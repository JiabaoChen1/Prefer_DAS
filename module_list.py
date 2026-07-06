import torch
import torch.nn.functional as F
import numpy as np
import os
import copy
import random
import cv2
import monai
import torch.nn as nn
from PIL import Image

from torch.optim.lr_scheduler import _LRScheduler
import torchvision.transforms.functional as transforms_f
from util.generate_heatmap import generate_gaussian


# --------------------------------------------------------------------------------
# Define EMA: Mean Teacher Framework
# --------------------------------------------------------------------------------
class EMA(object):
    def __init__(self, model, alpha):
        self.step = 0
        self.model = copy.deepcopy(model)
        self.alpha = alpha

    def update(self, model):
        decay = min(1 - 1 / (self.step + 1), self.alpha)
        for ema_param, param in zip(self.model.parameters(), model.parameters()):
            ema_param.data = decay * ema_param.data + (1 - decay) * param.data
        self.step += 1


# stage2的ema
class EMA_2(object):
    def __init__(self, model, alpha):
        self.step = 0
        self.model = copy.deepcopy(model)
        self.alpha = alpha

    def update(self, model):
        for ema_param, param in zip(self.model.parameters(), model.parameters()):
            ema_param.data = 0.99 * ema_param.data + (1 - 0.99) * param.data
        self.step += 1


# fixed ema
class EMA_fixed(object):
    def __init__(self, model, alpha):
        self.model = copy.deepcopy(model)
        self.alpha = alpha

    def update(self, model):
        for ema_param, param in zip(self.model.parameters(), model.parameters()):
            ema_param.data = self.alpha * ema_param.data + (1 - self.alpha) * param.data


# --------------------------------------------------------------------------------
# Define Polynomial Decay
# --------------------------------------------------------------------------------
class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1, min_lr=1e-6):
        self.power = power
        self.max_iters = max_iters
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [max(base_lr * (1 - self.last_epoch / self.max_iters) ** self.power, self.min_lr) for base_lr in self.base_lrs]

# --------------------------------------------------------------------------------
# Define training losses
# --------------------------------------------------------------------------------
def compute_supervised_loss(predict, target, reduction=True):
    # target[~mask_gt] = -1
    if reduction:
        loss = F.binary_cross_entropy(predict[:, 1, ::], target.float())
    else:
        loss = F.binary_cross_entropy(predict[:, 1, ::], target.float(), reduction="none")
    return loss


def compute_supervised_loss_ignore(predict, target, ignore_index=-1):
    mask = target != ignore_index
    target[target == ignore_index] = 0
    loss = F.binary_cross_entropy(predict[:, 1, ::], target.float(), reduction="none")
    loss = torch.sum(loss * mask) / torch.sum(mask)
    return loss


def compute_logprobs(logits, labels, mask=None):
    """
    logits: (B, 2, H, W) — 已过 softmax 的概率分布
    labels: (B, H, W) — 每像素为 0 或 1 /(B,N,H,W)
    mask:   (B, H, W) — 可选掩码
    """
    # 防止 log(0) 导致数值问题
    eps = 1e-8
    logps = torch.log(logits + eps)  # 已经是概率了，直接 log
    
    if(len(labels.shape) == 4):
        select_logprobs = torch.cat([torch.gather(logps, dim=1, index=labels[:,i,::].unsqueeze(1)) for i in range(labels.shape[1])],dim=1)

        if mask is not None:
            select_logprobs = select_logprobs * mask
            avg_logprob = select_logprobs.sum(dim=(2, 3)) / (mask.sum(dim=(2, 3)) + eps)
        else:
            avg_logprob = select_logprobs.mean(dim=(2, 3))

        return avg_logprob  # shape: (B,N)
    elif(len(labels.shape) == 3):
        labels = labels.unsqueeze(1)  # (B, 1, H, W)
        select_logprobs = torch.gather(logps, dim=1, index=labels).squeeze(1)  # (B, H, W)

        if mask is not None:
            select_logprobs = select_logprobs * mask
            avg_logprob = select_logprobs.sum(dim=(1, 2)) / (mask.sum(dim=(1, 2)) + eps)
        else:
            avg_logprob = select_logprobs.mean(dim=(1, 2))

        return avg_logprob  # shape: (B,)


class DPOLoss(nn.Module):
    """
    DPO Loss
    """

    def __init__(self, beta: float = 1) -> None:
        super().__init__()
        self.beta = beta

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ):
        """
        policy_chosen_logps: 模型输出的对数概率。Shape: (batch_size,)
        policy_rejected_logps:   Shape: (batch_size,)
        reference_chosen_logps: Shape: (batch_size,)
        reference_rejected_logps: Shape: (batch_size,)
        """
        policy_logps = policy_chosen_logps - policy_rejected_logps
        reference_logps = reference_chosen_logps - reference_rejected_logps
        logits = policy_logps - reference_logps

        loss = -F.logsigmoid(self.beta * logits)

        # 下面两个用于追踪训练的进度
        chosen_rewards = (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = (policy_rejected_logps - reference_rejected_logps).detach()

        # 对每个batch进行平均
        return loss.mean(), chosen_rewards.mean(), rejected_rewards.mean()

def compute_DPO(policy_predict, reference_predict, chosen_labels, rejected_labels,mask = None):
    loss_fn = DPOLoss()  # DPO loss
    policy_chosen_logps = compute_logprobs(policy_predict, chosen_labels, mask)
    policy_rejected_logps = compute_logprobs(policy_predict, rejected_labels, mask)
    reference_chosen_logps = compute_logprobs(reference_predict, chosen_labels, mask)
    reference_rejected_logps = compute_logprobs(reference_predict, rejected_labels, mask)
    loss, chosen_rewards, rejected_rewards = loss_fn(
        policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps
    )
    return loss

class SimPo(nn.Module):
    """
    SimPO Loss
    """

    def __init__(self, beta: float = 10, gamma_beta_ratio: float = 0.5) -> None:
        super().__init__()
        self.beta = beta
        self.gamma_beta_ratio = gamma_beta_ratio

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
    ):
        """
        policy_chosen_logps: 模型输出的对数概率。Shape: (batch_size,)
        policy_rejected_logps:   Shape: (batch_size,)
        """
        logits = policy_chosen_logps - policy_rejected_logps
        logits = logits - self.gamma_beta_ratio
        loss = -F.logsigmoid(self.beta * logits)

        # 对每个batch进行平均(期望)
        return loss.mean()

def compute_SimPo(policy_predict, chosen_labels, rejected_labels):
    loss_fn = SimPo()  # DPO loss
    policy_chosen_logps = compute_logprobs(policy_predict, chosen_labels)
    policy_rejected_logps = compute_logprobs(policy_predict, rejected_labels)
    loss = loss_fn(
        policy_chosen_logps, policy_rejected_logps
    )
    return loss

class SDPOLoss(nn.Module):
    """
    DPO Loss
    """

    def __init__(self, beta: float = 1) -> None:
        super().__init__()
        self.beta = beta

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ):
        """
        policy_chosen_logps: 模型输出的对数概率。Shape: (batch_size,)
        policy_rejected_logps:   Shape: (batch_size,)
        reference_chosen_logps: Shape: (batch_size,)
        reference_rejected_logps: Shape: (batch_size,)
        """
        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = []
        # for policy,reference in zip(policy_rejected_logps,reference_rejected_logps):
        #     rejected_logratios.append(policy - reference)
        if(len(policy_rejected_logps.shape) > 1):
            for i in range(policy_rejected_logps.shape[1]):
                rejected_logratios.append(policy_rejected_logps[:,i] - reference_rejected_logps[:,i])

            temp = sum(torch.exp(self.beta * (rejected - chosen_logratios)) for rejected in rejected_logratios)
            temp1 = -torch.log(temp)
            loss = -F.logsigmoid(temp1)
        else:
            for i in range(policy_rejected_logps.shape[0]):
                rejected_logratios.append(policy_rejected_logps[i] - reference_rejected_logps[i])

            temp = sum(torch.exp(self.beta * (rejected - chosen_logratios)) for rejected in rejected_logratios)
            temp1 = -torch.log(temp)
            loss = -F.logsigmoid(temp1)

        # 对每个batch进行平均
        return loss.mean()

def compute_SDPO(policy_predict, reference_predict, chosen_labels, rejected_labels,mask = None):
    loss_fn = SDPOLoss()  # DPO loss
    policy_chosen_logps = compute_logprobs(policy_predict, chosen_labels, mask)
    policy_rejected_logps = compute_logprobs(policy_predict, rejected_labels, mask)
    reference_chosen_logps = compute_logprobs(reference_predict, chosen_labels, mask)
    reference_rejected_logps = compute_logprobs(reference_predict, rejected_labels, mask)
    loss = loss_fn(
        policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps
    )
    return loss

def compute_iou(pred, target, eps=1e-6):
    # pred, target: (B, H, W), binary masks
    intersection = (pred & target).float().sum(dim=(1, 2))
    union = (pred | target).float().sum(dim=(1, 2))
    iou = (intersection + eps) / (union + eps)
    return iou  # shape: (B,)


def compute_unsupervised_loss(predict, target, logits, strong_threshold, weak_threshold):
    batch_size = predict.shape[0]
    valid_mask = (target >= 0).float()  # only count valid pixels
    # target[mask_gt] = -1
    # target[(target == 1) * (logits < weak_threshold)] = -1
    # target[(target == 0) * (logits < 0.6)] = -1
    weighting = logits.view(batch_size, -1).ge(strong_threshold).sum(-1) / valid_mask.view(batch_size, -1).sum(-1)
    # print('weighting{}'.format(weighting))
    loss = F.binary_cross_entropy(predict[:, 1, ::], target.float(), reduction="none")
    loss[(target == 1) * (logits < weak_threshold)] = 0
    loss[(target == 0) * (logits < weak_threshold)] = 0
    # if(cosine.sum() != 0):
    #     loss[(target == 1) * (cosine[:,0] < 0.5)] = 0
    # loss = F.cross_entropy(predict, target, reduction='none', ignore_index=-1)
    weighted_loss = torch.mean(torch.masked_select(weighting[:, None, None] * loss, loss > 0))
    return weighted_loss

def compute_pseudo_unsupervised_loss(predict, target):
    """
    predict: Tensor of shape [B, 2, H, W] (logits or softmaxed)
    target: Tensor of shape [B, H, W] with values 1 (positive) or -1 (ignore)
    """
    # 仅对 target == 1 的位置计算 BCE
    valid_mask = (target == 1).float()
    
    # 二分类时通常只取正类概率作为预测值
    # 如果 predict 是 softmax 后的概率，则用 predict[:, 1]
    # 如果是 logits，则改为 F.binary_cross_entropy_with_logits
    prob = predict[:, 1, :, :]  # shape [B, H, W]
    weight_map = prob.detach()  # 或 softmax 前景概率

    # 只在 target == 1 的地方计算损失（负类不存在）
    loss = F.binary_cross_entropy(prob, torch.ones_like(prob), reduction='none')
    loss = loss * valid_mask  # mask 掉 ignore 区域（target == -1）

    return loss.sum() / (valid_mask.sum() + 1e-8)

def compute_unsupervised_loss_det(predict, target, logits, detetion_point, strong_threshold, weak_threshold):
    batch_size = predict.shape[0]
    detection = detetion_point.cpu().bool()
    valid_mask = (target >= 0).float()  # only count valid pixels
    # target[(target == 1) * (logits < weak_threshold)] = -1
    # detection select foreground
    foreground = ((target == 1) * (logits > 0.9)).cpu().numpy().astype(np.uint8)
    detection_psedu = np.zeros_like(foreground)
    for j in range(batch_size):
        _, labels = cv2.connectedComponents(foreground[j])
        components = np.unique(labels[detection[j]])
        for i in components:
            if (i == 0) or ((labels == i).sum() > 256 * 256):
                continue
            detection_psedu[j][labels == i] = 1
    detection_psedu = torch.from_numpy(detection_psedu).to("cuda:0")
    # target[(target == 1) * (logits < 0.8)] = -1
    # target[(target == 1) * (0.8 < logits) * (logits < 0.97) * (detection_psedu == 0)] = -1
    # 生成可用区域
    mask = torch.zeros_like(target)
    mask[detection_psedu == 1] = 1
    mask[target == 0] = 1
    # mask[(target == 1)*(logits > 0.97)] = 1
    target[~mask.bool()] = -1
    weighting = logits.view(batch_size, -1).ge(strong_threshold).sum(-1) / valid_mask.view(batch_size, -1).sum(-1)
    # print('weighting{}'.format(weighting))
    loss = F.cross_entropy(predict, target, reduction="none", ignore_index=-1)
    weighted_loss = torch.mean(torch.masked_select(weighting[:, None, None] * loss, loss > 0))
    return weighted_loss



# --------------------------------------------------------------------------------
# Define ReCo loss
# --------------------------------------------------------------------------------
def compute_reco_loss(
    rep,
    label,
    mask,
    prob,
    positive_mask,
    query_mask,
    strong_threshold=1.0,
    temp=0.5,
    num_queries=256,
    num_negatives=256,
    preheat=False,
    epoch=0,
):
    batch_size, num_feat, im_w_, im_h = rep.shape
    num_segments = label.shape[1]
    device = rep.device

    # compute valid binary mask for each pixel
    valid_pixel = label * mask
    valid_positive_pixel = label * positive_mask
    valid_query_pixel = label * query_mask
    # permute representation for indexing: batch x im_h x im_w x feature_channel
    rep = rep.permute(0, 2, 3, 1)
    seg_feat_negative_list = []
    seg_feat_hard_list = []
    seg_feat_hard_hard_list = []
    seg_num_list = []
    # compute prototype (class mean representation) for each class across all valid pixels
    if preheat:
        seg_proto_list = []
        for i in range(num_segments):
            valid_pixel_seg = valid_pixel[:, i]  # select binary mask for i-th class
            valid_query_pixel_seg = valid_query_pixel[:, i, :, :].bool()
            valid_positive_pixel_seg = valid_pixel_seg
            if valid_pixel_seg.sum() == 0:  # not all classes would be available in a mini-batch
                print("{}类无可用pixel".format(i))
                continue
            # if valid_positive_pixel_seg.sum() == 0:  # not all classes would be available in a mini-batch
            #     print('{}类无可用positve pixel'.format(i))
            if valid_query_pixel_seg.sum() == 0:  # not all classes would be available in a mini-batch
                print("{}类无可用query pixel".format(i))
                continue

            prob_seg = prob[:, i, :, :]
            rep_mask_hard = (prob_seg < strong_threshold) * valid_pixel_seg.bool()  # select hard queries
            # select query
            seg_proto_list.append(torch.mean(rep[valid_positive_pixel_seg.bool()], dim=0, keepdim=True))  # positve
            seg_feat_negative_list.append(rep[valid_pixel_seg.bool()])  # negative
            seg_feat_hard_list.append(rep[valid_query_pixel_seg])  # query
            seg_feat_hard_hard_list.append(rep[rep_mask_hard])  # 难query
            seg_num_list.append(int(valid_pixel_seg.sum().item()))  # 全部可用pixel数量

        # if(len(torch.unique(valid_positive_pixel[:, 0, :, :])) < 2):
        #     print('背景类无有标签pixel')
        #     return torch.tensor(0.0)
        # seg_feat_hard_list[0] = rep[valid_positive_pixel[:, 0, :, :].bool()]
        # compute regional contrastive loss
        if len(seg_num_list) <= 1:  # in some rare cases, a small mini-batch might only contain 1 or no semantic class
            print("mini-batch只有一类,对比损失为0")
            return torch.tensor(0.0)
        else:
            reco_loss = torch.tensor(0.0)
            seg_proto = torch.cat(seg_proto_list)  # positve
            valid_seg = len(seg_num_list)

            i = 1
            # sample hard queries
            if epoch < 115:
                if (
                    len(seg_feat_hard_list[i]) > 0 and valid_positive_pixel[:, 1, :, :].sum() != 0
                ):  # 当hard query数量大于0,随机抽取num_queries个
                    seg_hard_idx = torch.randint(len(seg_feat_hard_list[i]), size=(num_queries,))
                    anchor_feat_hard = seg_feat_hard_list[i][seg_hard_idx]
                    anchor_feat = anchor_feat_hard
                else:  # in some rare cases, all queries in the current query class are easy
                    return torch.tensor(0.0)
            else:
                if (
                    len(seg_feat_hard_hard_list[i]) > 0 and len(seg_feat_hard_list[i]) > 0 and valid_positive_pixel[:, 1, :, :].sum() != 0
                ):  # 当hard query数量大于0,随机抽取num_queries个
                    seg_hard_idx = torch.randint(len(seg_feat_hard_list[i]), size=(num_queries,))
                    seg_hard_hard_idx = torch.randint(len(seg_feat_hard_hard_list[i]), size=(num_queries,))
                    anchor_feat_hard = seg_feat_hard_list[i][seg_hard_idx]
                    anchor_feat_hard_hard = seg_feat_hard_hard_list[i][seg_hard_hard_idx]
                    anchor_feat = anchor_feat_hard + anchor_feat_hard_hard
                else:  # in some rare cases, all queries in the current query class are easy
                    return torch.tensor(0.0)
            # if (len(seg_feat_hard_list[i]) > 0 and valid_positive_pixel[:, 1, :, :].sum() != 0):#当hard query数量大于0,随机抽取num_queries个
            #     seg_hard_idx = torch.randint(len(seg_feat_hard_list[i]), size=(num_queries,))
            #     anchor_feat_hard = seg_feat_hard_list[i][seg_hard_idx]
            #     anchor_feat = anchor_feat_hard
            # else:  # in some rare cases, all queries in the current query class are easy
            #     return torch.tensor(0.0)
            # apply negative key sampling (with no gradients)
            with torch.no_grad():
                # 可抽样样本数
                negative_num_valid = seg_num_list[0]
                if negative_num_valid < num_negatives:
                    print("类{} negative{},不足{}".format(i, negative_num_valid, num_negatives))
                    return torch.tensor(0.0)
                # 有放回采样
                negative_index = np.random.randint(low=0, high=negative_num_valid, size=(num_queries * num_negatives)).tolist()

                # index negative keys (from other classes)
                negative_feat_all = seg_feat_negative_list[0]
                negative_feat = negative_feat_all[negative_index].reshape(num_queries, num_negatives, num_feat)

                # combine positive and negative keys: keys = [positive key | negative keys] with 1 + num_negative dim
                positive_feat = seg_proto[i].unsqueeze(0).unsqueeze(0).repeat(num_queries, 1, 1)

                all_feat = torch.cat((positive_feat, negative_feat), dim=1)

            seg_logits1 = torch.cosine_similarity(anchor_feat.unsqueeze(1), all_feat, dim=2)

            reco_loss = reco_loss + F.cross_entropy(seg_logits1 / temp, torch.zeros(num_queries).long().to(device))

            # i = 0

            # # sample hard queries
            # if len(seg_feat_hard_list[i]) > 0:#当hard query数量大于0,随机抽取num_queries个
            #     seg_hard_idx = torch.randint(len(seg_feat_hard_list[i]), size=(num_queries,))
            #     anchor_feat_hard = seg_feat_hard_list[i][seg_hard_idx]
            #     anchor_feat = anchor_feat_hard
            # else:  # in some rare cases, all queries in the current query class are easy
            #     return torch.tensor(0.0)

            # # apply negative key sampling (with no gradients)
            # with torch.no_grad():

            #     # # index negative keys (from other classes)
            #     # negative_feat_back = positive_feat.repeat(1,num_negatives, 1)

            #     # # combine positive and negative keys: keys = [positive key | negative keys] with 1 + num_negative dim
            #     # positive_feat = torch.mean(negative_feat,dim = 1, keepdim=True)

            #     # all_feat = torch.cat((positive_feat, negative_feat_back), dim=1)

            #     # sampling negative keys based on the generated distribution [num_queries x num_negatives]
            #     negative_dist1 = torch.distributions.categorical.Categorical(probs=torch.tensor([1]).to(device))
            #     samp_class1 = negative_dist1.sample(sample_shape=[num_queries, num_negatives])
            #     samp_num1 = torch.stack([(samp_class1 == c).sum(1) for c in range(len([1]))], dim=1)

            #     # 可抽样样本数
            #     negative_num_list = seg_num_list[i+1:] + seg_num_list[:i]
            #     if(negative_num_list[0] < num_negatives):
            #         print('类{} negative{},不足{}'.format(i,negative_num_list[0],num_negatives))
            #         return torch.tensor(0.0)
            #     #有放回采样
            #     negative_index = negative_index_sampler(samp_num1, negative_num_list)

            #     # index negative keys (from other classes)
            #     negative_feat_all = torch.cat(seg_feat_all_list[i+1:] + seg_feat_all_list[:i])
            #     negative_feat = negative_feat_all[negative_index].reshape(num_queries, num_negatives, num_feat)

            #     # combine positive and negative keys: keys = [positive key | negative keys] with 1 + num_negative dim
            #     positive_feat = seg_proto[i].unsqueeze(0).unsqueeze(0).repeat(num_queries, 1, 1)

            #     all_feat = torch.cat((positive_feat, negative_feat), dim=1)
            # seg_logits1 = torch.cosine_similarity(anchor_feat.unsqueeze(1), all_feat, dim=2)

            # reco_loss = reco_loss + F.cross_entropy(seg_logits1 / temp, torch.zeros(num_queries).long().to(device))
            return reco_loss
    else:
        seg_proto_list = []
        for i in range(num_segments):
            valid_pixel_seg = valid_pixel[:, i]  # select binary mask for i-th class
            if valid_pixel_seg.sum() == 0:  # not all classes would be available in a mini-batch
                print("{}类无可用pixel".format(i))
                continue

            prob_seg = prob[:, i, :, :]
            rep_mask_hard = (prob_seg < strong_threshold) * valid_pixel_seg.bool()  # select hard queries
            # rep_mask_hard = valid_positive_pixel[:, i, :, :].bool()
            seg_proto_list.append(torch.mean(rep[valid_pixel_seg.bool()], dim=0, keepdim=True))
            seg_feat_negative_list.append(rep[valid_pixel_seg.bool()])  # 全部可用pixel
            seg_feat_hard_list.append(rep[rep_mask_hard])  # 难query
            seg_num_list.append(int(valid_pixel_seg.sum().item()))  # 全部可用pixel数量

        # if(len(torch.unique(valid_positive_pixel[:, 0, :, :])) < 2):
        #     print('背景类无有标签pixel')
        #     return torch.tensor(0.0)
        # seg_feat_hard_list[0] = rep[valid_positive_pixel[:, 0, :, :].bool()]
        # compute regional contrastive loss
        if len(seg_num_list) <= 1:  # in some rare cases, a small mini-batch might only contain 1 or no semantic class
            print("mini-batch只有一类,对比损失为0")
            return torch.tensor(0.0)
        else:
            reco_loss = torch.tensor(0.0)
            seg_proto = torch.cat(seg_proto_list)  # positve
            valid_seg = len(seg_num_list)
            seg_len = torch.arange(valid_seg)

            i = 1
            # sample hard queries
            if len(seg_feat_hard_list[i]) > 0:  # 当hard query数量大于0,随机抽取num_queries个
                seg_hard_idx = torch.randint(len(seg_feat_hard_list[i]), size=(num_queries,))
                anchor_feat_hard = seg_feat_hard_list[i][seg_hard_idx]
                anchor_feat = anchor_feat_hard
            else:  # in some rare cases, all queries in the current query class are easy
                print("{}类query数量为{}不够".format(i, len(seg_feat_hard_list[i])))
                return torch.tensor(0.0)

            # apply negative key sampling (with no gradients)
            with torch.no_grad():
                # sampling negative keys based on the generated distribution [num_queries x num_negatives]
                negative_dist1 = torch.distributions.categorical.Categorical(probs=torch.tensor([1]).to(device))
                samp_class1 = negative_dist1.sample(sample_shape=[num_queries, num_negatives])
                samp_num1 = torch.stack([(samp_class1 == c).sum(1) for c in range(len([1]))], dim=1)

                # 可抽样样本数
                negative_num_list = seg_num_list[i + 1 :] + seg_num_list[:i]
                if negative_num_list[0] < num_negatives:
                    print("类{} negative{},不足{}".format(negative_num_list[0], i, num_negatives))
                    return torch.tensor(0.0)
                # 有放回采样
                negative_index = negative_index_sampler(samp_num1, negative_num_list)

                # index negative keys (from other classes)
                negative_feat_all = torch.cat(seg_feat_negative_list[i + 1 :] + seg_feat_negative_list[:i])
                negative_feat = negative_feat_all[negative_index].reshape(num_queries, num_negatives, num_feat)

                # combine positive and negative keys: keys = [positive key | negative keys] with 1 + num_negative dim
                positive_feat = seg_proto[i].unsqueeze(0).unsqueeze(0).repeat(num_queries, 1, 1)

                all_feat = torch.cat((positive_feat, negative_feat), dim=1)

            seg_logits1 = torch.cosine_similarity(anchor_feat.unsqueeze(1), all_feat, dim=2)

            reco_loss = reco_loss + F.cross_entropy(seg_logits1 / temp, torch.zeros(num_queries).long().to(device))

            i = 0

            # sample hard queries
            if len(seg_feat_hard_list[i]) > 0:  # 当hard query数量大于0,随机抽取num_queries个
                seg_hard_idx = torch.randint(len(seg_feat_hard_list[i]), size=(num_queries,))
                anchor_feat_hard = seg_feat_hard_list[i][seg_hard_idx]
                anchor_feat = anchor_feat_hard
            else:  # in some rare cases, all queries in the current query class are easy
                print("{}类query数量为{}不够".format(i, len(seg_feat_hard_list[i])))
                return torch.tensor(0.0)

            # apply negative key sampling (with no gradients)
            with torch.no_grad():

                # # index negative keys (from other classes)
                # negative_feat_back = positive_feat.repeat(1,num_negatives, 1)

                # # combine positive and negative keys: keys = [positive key | negative keys] with 1 + num_negative dim
                # positive_feat = torch.mean(negative_feat,dim = 1, keepdim=True)

                # all_feat = torch.cat((positive_feat, negative_feat_back), dim=1)

                # sampling negative keys based on the generated distribution [num_queries x num_negatives]
                negative_dist1 = torch.distributions.categorical.Categorical(probs=torch.tensor([1]).to(device))
                samp_class1 = negative_dist1.sample(sample_shape=[num_queries, num_negatives])
                samp_num1 = torch.stack([(samp_class1 == c).sum(1) for c in range(len([1]))], dim=1)

                # 可抽样样本数
                negative_num_list = seg_num_list[i + 1 :] + seg_num_list[:i]
                if negative_num_list[0] < num_negatives:
                    print("类{} negative{},不足{}".format(negative_num_list[0], i, num_negatives))
                    return torch.tensor(0.0)
                # 有放回采样
                negative_index = negative_index_sampler(samp_num1, negative_num_list)

                # index negative keys (from other classes)
                negative_feat_all = torch.cat(seg_feat_negative_list[i + 1 :] + seg_feat_negative_list[:i])
                negative_feat = negative_feat_all[negative_index].reshape(num_queries, num_negatives, num_feat)

                # combine positive and negative keys: keys = [positive key | negative keys] with 1 + num_negative dim
                positive_feat = seg_proto[i].unsqueeze(0).unsqueeze(0).repeat(num_queries, 1, 1)

                all_feat = torch.cat((positive_feat, negative_feat), dim=1)
            seg_logits1 = torch.cosine_similarity(anchor_feat.unsqueeze(1), all_feat, dim=2)

            reco_loss = reco_loss + F.cross_entropy(seg_logits1 / temp, torch.zeros(num_queries).long().to(device))
            return reco_loss / valid_seg


def negative_index_sampler(samp_num, seg_num_list):
    negative_index = []
    for i in range(samp_num.shape[0]):
        for j in range(samp_num.shape[1]):
            negative_index += np.random.randint(
                low=sum(seg_num_list[:j]), high=sum(seg_num_list[: j + 1]), size=int(samp_num[i, j])
            ).tolist()
    return negative_index


# --------------------------------------------------------------------------------
# Define evaluation metrics
# --------------------------------------------------------------------------------
class ConfMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.jac = 0
        self.dice = 0
        self.num = 0
        self.compute_dice = monai.metrics.DiceMetric(include_background=False, ignore_empty=False)

    def update(self, pred, target):
        binary = torch.nn.functional.one_hot(target, num_classes=2).permute(0, 3, 1, 2)
        pre = torch.nn.functional.one_hot(pred.argmax(1), num_classes=2).permute(0, 3, 1, 2)
        jac = monai.metrics.compute_iou(pre, binary, include_background=False, ignore_empty=False)
        dice = self.compute_dice(pre, binary)
        self.jac += torch.mean(jac)
        self.dice += torch.mean(dice)
        self.num += 1

    def get_jac(self):
        return self.jac / self.num

    def get_dice(self):
        return self.dice / self.num


# --------------------------------------------------------------------------------
# Define useful functions
# --------------------------------------------------------------------------------
def label_binariser(inputs):
    outputs = torch.zeros_like(inputs).to(inputs.device)
    index = torch.max(inputs, dim=1)[1]
    outputs = outputs.scatter_(1, index.unsqueeze(1), 1.0)
    return outputs


def label_onehot(inputs, num_segments):
    batch_size, im_h, im_w = inputs.shape
    # remap invalid pixels (-1) into 0, otherwise we cannot create one-hot vector with negative labels.
    # we will still mask out those invalid values in valid mask
    inputs = torch.relu(inputs)
    outputs = torch.zeros([batch_size, num_segments, im_h, im_w]).to(inputs.device)
    return outputs.scatter_(1, inputs.unsqueeze(1), 1.0)


def denormalise(x, imagenet=True):
    if imagenet:
        x = transforms_f.normalize(x, mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
        x = transforms_f.normalize(x, mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0])
        return x
    else:
        return (x + 1) / 2


def create_folder(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


def tensor_to_pil(im, label, logits, point=None):
    im = denormalise(im)
    im = transforms_f.to_pil_image(im.cpu())

    label = transforms_f.to_pil_image((label.float() / 255.0).unsqueeze(0).cpu())
    logits = transforms_f.to_pil_image(logits.unsqueeze(0).cpu())

    outputs = [im, label, logits]

    if point is not None:
        point = transforms_f.to_pil_image(point.unsqueeze(0).cpu())
        outputs.append(point)


    return tuple(outputs)


def tensor_to_pil_sam(im, label, sam_label, logits, point=None):
    im = denormalise(im)
    im = transforms_f.to_pil_image(im.cpu())

    label = label.float() / 255.0
    label = transforms_f.to_pil_image(label.unsqueeze(0).cpu())

    sam_label = sam_label.float() / 255.0
    sam_label = transforms_f.to_pil_image(sam_label.unsqueeze(0).cpu())

    logits = transforms_f.to_pil_image(logits.unsqueeze(0).cpu())
    if point is not None:
        point = transforms_f.to_pil_image(point.unsqueeze(0).cpu())
        return im, label, sam_label, logits, point
    else:
        return im, label, sam_label, logits


# --------------------------------------------------------------------------------
# Define semi-supervised methods (based on data augmentation)
# --------------------------------------------------------------------------------
def generate_cutout_mask(img_size, ratio=2):
    cutout_area = img_size[0] * img_size[1] / ratio

    w = np.random.randint(img_size[1] / ratio + 1, img_size[1])
    h = np.round(cutout_area / w)

    x_start = np.random.randint(0, img_size[1] - w + 1)
    y_start = np.random.randint(0, img_size[0] - h + 1)

    x_end = int(x_start + w)
    y_end = int(y_start + h)

    mask = torch.ones(img_size)
    mask[y_start:y_end, x_start:x_end] = 0
    return mask.float()


def generate_class_mask(pseudo_labels):
    labels = torch.unique(pseudo_labels)  # all unique labels
    labels_select = labels[torch.randperm(len(labels))][: len(labels) // 2]  # randomly select half of labels

    mask = (pseudo_labels.unsqueeze(-1) == labels_select).any(-1)
    return mask.float()


def generate_unsup_data(data, target, logits, point, mode="cutout"):
    batch_size, _, im_h, im_w = data.shape
    device = data.device

    new_data = []
    new_target = []
    new_logits = []
    new_point = []
    if point is not None:
        for i in range(batch_size):
            if mode == "cutout":
                mix_mask = generate_cutout_mask([im_h, im_w], ratio=2).to(device)
                target[i][(1 - mix_mask).bool()] = -1

                new_data.append((data[i] * mix_mask).unsqueeze(0))
                new_target.append(target[i].unsqueeze(0))
                new_logits.append((logits[i] * mix_mask).unsqueeze(0))
                new_point.append((point[i].unsqueeze(0)))
                continue
            if mode == "cutmix":
                mix_mask = generate_cutout_mask([im_h, im_w]).to(device)
            if mode == "classmix":
                mix_mask = generate_class_mask(target[i]).to(device)

            new_data.append((data[i] * mix_mask + data[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))
            new_target.append((target[i] * mix_mask + target[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))
            new_logits.append((logits[i] * mix_mask + logits[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))
            new_point.append((point[i] * mix_mask + point[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))

        new_data, new_target, new_logits, new_point = (
            torch.cat(new_data),
            torch.cat(new_target),
            torch.cat(new_logits),
            torch.cat(new_point),
        )
        return new_data, new_target.long(), new_logits, new_point
    else:
        for i in range(batch_size):
            if mode == "cutout":
                mix_mask = generate_cutout_mask([im_h, im_w], ratio=2).to(device)
                target[i][(1 - mix_mask).bool()] = -1

                new_data.append((data[i] * mix_mask).unsqueeze(0))
                new_target.append(target[i].unsqueeze(0))
                new_logits.append((logits[i] * mix_mask).unsqueeze(0))
                continue

            if mode == "cutmix":
                mix_mask = generate_cutout_mask([im_h, im_w]).to(device)
            if mode == "classmix":
                mix_mask = generate_class_mask(target[i]).to(device)

            new_data.append((data[i] * mix_mask + data[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))
            new_target.append((target[i] * mix_mask + target[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))
            new_logits.append((logits[i] * mix_mask + logits[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))

        new_data, new_target, new_logits = torch.cat(new_data), torch.cat(new_target), torch.cat(new_logits)
        return new_data, new_target.long(), new_logits
