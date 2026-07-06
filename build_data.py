from torch.utils.data.dataset import Dataset
from PIL import Image
from PIL import ImageFilter
import pandas as pd
import numpy as np
import torch
import os
import random
import glob
import cv2
import re

import torch.utils.data.sampler as sampler
import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_f

import albumentations as A

from module_list import *
from util.generate_heatmap import img_generate_gaussian, img_generate_centerpoint,pointmap_generate_gaussian
from util.point_prompt import get_point_prompt_heatmap, heatmap_label2prompt_heatmap, heatmap_label2prompt_heatmap_Focus


# --------------------------------------------------------------------------------
# Define data augmentation
# --------------------------------------------------------------------------------
cvlab_strong_aug = [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.GaussNoise(p=0.3),
                    A.ColorJitter(
                        brightness=[0.7, 1.3], contrast=[0.7, 1.3],
                        saturation=0.5, hue=[0, 0.5], p=0.5,
                    ),
                    A.MotionBlur(),
                    A.Perspective(p=0.3),
]
def transform(
    image,
    label,
    logits=None,
    point=None,
    pseudo=None,
    crop_size=(512, 512),
    scale_size=(0.8, 1.0),
    augmentation=True,
    weak=None,
    mic=False,
):
    raw_w, raw_h = image.size
    scale_ratio = random.uniform(scale_size[0], scale_size[1])
    resized_size = (int(raw_h * scale_ratio), int(raw_w * scale_ratio))

    # Resize
    image = transforms_f.resize(image, resized_size, Image.BILINEAR)
    label = transforms_f.resize(label, resized_size, Image.NEAREST)
    if logits is not None:
        logits = transforms_f.resize(logits, resized_size, Image.NEAREST)
    if point is not None:
        point = transforms_f.resize(point, resized_size, Image.NEAREST)
    if pseudo is not None:
        pseudo = transforms_f.resize(pseudo, resized_size, Image.NEAREST)
    # Padding if needed
    if crop_size == -1:
        crop_size = (raw_h, raw_w)
    pad_h = max(crop_size[0] - resized_size[0], 0)
    pad_w = max(crop_size[1] - resized_size[1], 0)
    if pad_h > 0 or pad_w > 0:
        pad = (0, 0, pad_w, pad_h)
        image = transforms_f.pad(image, pad, padding_mode="reflect")
        label = transforms_f.pad(label, pad, fill=255, padding_mode="constant")
        if logits is not None:
            logits = transforms_f.pad(logits, pad, fill=0, padding_mode="constant")
        if point is not None:
            point = transforms_f.pad(point, pad, fill=0, padding_mode="constant")
        if pseudo is not None:
            pseudo = transforms_f.pad(pseudo, pad, fill=0, padding_mode="constant")

    # Compose input dict
    input_dict = {"image": np.array(image), "mask": np.array(label)}
    if logits is not None:
        input_dict["logit"] = np.array(logits)
    if point is not None:
        input_dict["point"] = np.array(point)
    if pseudo is not None:
        input_dict["pseudo"] = np.array(pseudo)

    # Crop transforms
    crop_transform = A.Compose(
        [
            A.CropNonEmptyMaskIfExists(crop_size[0], crop_size[1], ignore_values=[255], p=0.7),
            A.RandomCrop(crop_size[0], crop_size[1]),
        ],
        additional_targets={
            "point": "mask", "logit": "mask", "pseudo": "mask"
        },
    )
    input_dict = crop_transform(**input_dict)

    # Augmentation transforms
    if augmentation:
        if weak is None:
            aug_transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.GaussNoise(p=0.3),
                    A.ColorJitter(
                        brightness=[0.7, 1.3], contrast=[0.7, 1.3],
                        saturation=0.5, hue=[0, 0.5], p=0.5,
                    ),
                    A.Perspective(p=0.3),
                ],
                # cvlab_strong_aug,
                additional_targets={
                    "point": "mask", "logit": "mask"
                },
            )
        else:
            aug_transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                ],
                additional_targets={
                    "point": "mask", "logit": "mask", "pseudo": "mask"
                },
            )
        input_dict = aug_transform(**input_dict)

    # Helper to convert to PIL
    def to_pil(key):
        return Image.fromarray(input_dict[key]) if key in input_dict else None

    image = to_pil("image")
    label = to_pil("mask")
    point = to_pil("point")
    logits = to_pil("logit")
    pseudo = to_pil("pseudo")

    # To tensor
    image = transforms_f.to_tensor(image)
    label = (transforms_f.to_tensor(label) * 255).long()
    label[label == 255] = -1
    if logits is not None:
        logits = transforms_f.to_tensor(logits)
    if point is not None:
        point = torch.from_numpy(np.array(point)).float()
    if pseudo is not None:
        pseudo = torch.from_numpy(np.array(pseudo)).float()

    # Normalize
    image = transforms_f.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Compose return tuple
    result = [image, label]
    if logits is not None:
        result.append(logits)
    if point is not None:
        result.append(point)
    if pseudo is not None:
        result.append(pseudo)
    return tuple(result)


def transform_DPO(
    image,
    label,
    logits=None,
    point=None,
    crop_size=(512, 512),
    scale_size=(0.8, 1.0),
    augmentation=True,
    weak=None,
    mic=False,
):
    # Random rescale image
    raw_w, raw_h = image.size
    scale_ratio = random.uniform(scale_size[0], scale_size[1])

    resized_size = (int(raw_h * scale_ratio), int(raw_w * scale_ratio))
    image = transforms_f.resize(image, resized_size, Image.BILINEAR)
    label = transforms_f.resize(label, resized_size, Image.NEAREST)
    if logits is not None:
        logits = transforms_f.resize(logits, resized_size, Image.NEAREST)
    if point is not None:
        point = transforms_f.resize(point, resized_size, Image.NEAREST)

    # Add padding if rescaled image size is less than crop size
    if crop_size == -1:  # use original im size without crop or padding
        crop_size = (raw_h, raw_w)

    if crop_size[0] > resized_size[0] or crop_size[1] > resized_size[1]:
        right_pad, bottom_pad = max(crop_size[1] - resized_size[1], 0), max(crop_size[0] - resized_size[0], 0)
        image = transforms_f.pad(image, padding=(0, 0, right_pad, bottom_pad), padding_mode="reflect")
        label = transforms_f.pad(label, padding=(0, 0, right_pad, bottom_pad), fill=255, padding_mode="constant")
        if logits is not None:
            logits = transforms_f.pad(logits, padding=(0, 0, right_pad, bottom_pad), fill=0, padding_mode="constant")
        if point is not None:
            point = transforms_f.pad(point, padding=(0, 0, right_pad, bottom_pad), fill=0, padding_mode="constant")

    transform = A.Compose(
        [
            A.CropNonEmptyMaskIfExists(crop_size[0], crop_size[1], ignore_values=[255], p=0.7),
            A.RandomCrop(crop_size[0], crop_size[1]),
        ],
        additional_targets={"point": "mask", "logit": "image"},
    )

    # 3. 进行图像和掩码的裁剪
    if logits is not None:
        if point is not None:
            transformed = transform(image=np.array(image), mask=np.array(label), point=np.array(point), logit=np.array(logits))
            image = transformed["image"]
            label = transformed["mask"]
            point = transformed["point"]
            logits = transformed["logit"]
        else:
            transformed = transform(image=np.array(image), mask=np.array(label), logit=np.array(logits))
            image = transformed["image"]
            label = transformed["mask"]
            logits = transformed["logit"]
    elif point is not None:
        transformed = transform(image=np.array(image), mask=np.array(label), point=np.array(point))
        image = transformed["image"]
        label = transformed["mask"]
        point = transformed["point"]
    else:
        transformed = transform(image=np.array(image), mask=np.array(label))
        # 获取裁剪后的图像和掩码
        image = transformed["image"]
        label = transformed["mask"]

    if augmentation:
        if weak == None:
            aug = A.Compose(
                [
                    A.Flip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.GaussNoise(p=0.3),
                    A.ColorJitter(brightness=[0.7, 1.3], contrast=[0.7, 1.3], saturation=0.5, hue=[0, 0.5], p=0.5),
                    A.Perspective(p=0.3),
                ],
                additional_targets={"point": "mask", "logit": "mask"},
            )
        else:
            aug = A.Compose(
                [
                    A.Flip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    # A.GaussNoise(p=0.3),
                    # A.ColorJitter(brightness=[0.7, 1.3], contrast=[0.7, 1.3], saturation=0.5, hue=[0, 0.5],p=0.5),
                    # A.ElasticTransform(p=0.3),
                    # A.Perspective(p=0.3),
                ],
                additional_targets={"point": "mask", "logit": "mask"},
            )
        if logits is not None:
            if point is not None:
                transformed = aug(image=np.array(image), mask=np.array(label), point=np.array(point), logit=np.array(logits))
                image = Image.fromarray(transformed["image"])
                label = Image.fromarray(transformed["mask"])
                point = Image.fromarray(transformed["point"])
                logits = Image.fromarray(transformed["logit"])
            else:
                transformed = aug(image=np.array(image), mask=np.array(label), logit=np.array(logits))
                image = Image.fromarray(transformed["image"])
                label = Image.fromarray(transformed["mask"])
                logits = Image.fromarray(transformed["logit"])
        elif point is not None:
            transformed = aug(image=np.array(image), mask=np.array(label), point=np.array(point))
            image = Image.fromarray(transformed["image"])
            label = Image.fromarray(transformed["mask"])
            point = Image.fromarray(transformed["point"])
        else:
            transformed = aug(image=np.array(image), mask=np.array(label))
            image = Image.fromarray(transformed["image"])
            label = Image.fromarray(transformed["mask"])
        # Random color jitter
        # if torch.rand(1) > 0.5:
        #     color_transform = transforms.ColorJitter((0.75, 1.25), (0.75, 1.25), (0.75, 1.25), (-0.25, 0.25))  #For PyTorch 1.9/TorchVision 0.10 users
        #     image = color_transform(image)

        # if torch.rand(1) > 0.5:
        #     color_transform = transforms
        #     image = color_transform(image)
        # Random Gaussian filter
        # if torch.rand(1) > 0.5:
        #     sigma = random.uniform(0.15, 1.15)
        #     image = image.filter(ImageFilter.GaussianBlur(radius=sigma))

        # Random horizontal flipping
        # if torch.rand(1) > 0.5:
        #     image = transforms_f.hflip(image)
        #     label = transforms_f.hflip(label)
        #     if logits is not None:
        #         logits = transforms_f.hflip(logits)
        #     if point is not None:
        #         point = transforms_f.hflip(point)

    # Transform to tensor
    image = transforms_f.to_tensor(image)
    label = (transforms_f.to_tensor(label) * 255).long()
    label[label == 255] = -1  # invalid pixels are re-mapped to index -1
    if logits is not None:
        logits = transforms_f.to_tensor(logits)
    if point is not None:
        point = torch.from_numpy(np.array(point)).float()
        # point = transforms_f.to_tensor(point)

    # Apply (ImageNet) normalisation
    image = transforms_f.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if logits is not None:
        if point is not None:
            return image, label, logits, point
        return image, label, logits
    if point is not None:
        return image, label, point
    else:
        return image, label

def batch_transform(
    data, label, logits, point=None,crop_size=(512, 512), scale_size=(0.8, 1.0), apply_augmentation=True
):
    device = data.device
    B = data.shape[0]

    data_list, label_list, logits_list = [], [], []
    point_list, pseudo_list = [], []

    for k in range(B):
        # 转换为 PIL，按需返回 point 和 pseudo
        pil_outputs = tensor_to_pil(
            data[k], label[k], logits[k],
            point=point[k] if point is not None else None,
        )

        # 解包 PIL 数据
        im_pil, label_pil, logits_pil = pil_outputs[:3]
        point_pil = pil_outputs[3] if point is not None else None

        # 应用 transform
        transform_outputs = transform(
            im_pil, label_pil, logits_pil,
            point=point_pil,crop_size=crop_size, scale_size=scale_size, augmentation=apply_augmentation
        )

        aug_data, aug_label, aug_logits = transform_outputs[:3]
        aug_point = transform_outputs[3] if point is not None else None

        # 收集结果
        data_list.append(aug_data.unsqueeze(0))
        label_list.append(aug_label)
        logits_list.append(aug_logits)
        if point is not None:
            point_list.append(aug_point.unsqueeze(0))
    # 拼接结果
    output = [
        torch.cat(data_list).to(device),
        torch.cat(label_list).to(device),
        torch.cat(logits_list).to(device)
    ]
    if point is not None:
        output.append(torch.cat(point_list).to(device))

    return tuple(output)


def batch_transform_det(data, label, logits, point, crop_size, scale_size, apply_augmentation):
    data_list, label_list, logits_list, point_list = [], [], [], []
    device = data.device
    if point is not None:
        for k in range(data.shape[0]):
            data_pil, label_pil, logits_pil, point_pil = tensor_to_pil(data[k], label[k], logits[k], point[k])
            aug_data, aug_label, aug_logits, aug_point = transform(
                data_pil,
                label_pil,
                logits_pil,
                point_pil,
                crop_size=crop_size,
                scale_size=scale_size,
                augmentation=apply_augmentation,
                det_aug=True,
            )
            # aug_data, aug_label, aug_logits, aug_point = transform(data_pil, label_pil, logits_pil, point_pil,
            #                                             crop_size=crop_size,
            #                                             scale_size=scale_size,
            #                                             augmentation=apply_augmentation)
            data_list.append(aug_data.unsqueeze(0))
            label_list.append(aug_label)
            logits_list.append(aug_logits)
            point_list.append(aug_point.unsqueeze(0))

        data_trans, label_trans, logits_trans, point_trans = (
            torch.cat(data_list).to(device),
            torch.cat(label_list).to(device),
            torch.cat(logits_list).to(device),
            torch.cat(point_list).to(device),
        )
        return data_trans, label_trans, logits_trans, point_trans
    else:
        for k in range(data.shape[0]):
            data_pil, label_pil, logits_pil = tensor_to_pil(data[k], label[k], logits[k])
            aug_data, aug_label, aug_logits = transform(
                data_pil,
                label_pil,
                logits_pil,
                crop_size=crop_size,
                scale_size=scale_size,
                augmentation=apply_augmentation,
            )
            data_list.append(aug_data.unsqueeze(0))
            label_list.append(aug_label)
            logits_list.append(aug_logits)

        data_trans, label_trans, logits_trans = (
            torch.cat(data_list).to(device),
            torch.cat(label_list).to(device),
            torch.cat(logits_list).to(device),
        )
        return data_trans, label_trans, logits_trans


# --------------------------------------------------------------------------------
# Define segmentation label re-mapping
# --------------------------------------------------------------------------------


def k_class_map(mask):
    # 255为未确定的
    mask_map = np.zeros_like(mask)
    mask_map[np.isin(mask, [255])] = 1  # 线粒体
    mask_map[np.isin(mask, [0])] = 255  # 未确定的
    mask_map[np.isin(mask, [128])] = 0  # 背景
    return mask_map


def cvlab_class_map(mask):
    # 255为未确定的
    mask_map = np.zeros_like(mask)
    mask_map[np.isin(mask, [255])] = 1  # 线粒体
    mask_map[np.isin(mask, [0])] = 255  # 未确定的
    mask_map[np.isin(mask, [128])] = 0  # 背景
    return mask_map


def R_class_map(mask):
    # 255为未确定的
    mask_map = np.zeros_like(mask)
    mask_map[np.isin(mask, [255])] = 1  # 线粒体
    mask_map[np.isin(mask, [0])] = 255  # 未确定的
    mask_map[np.isin(mask, [128])] = 0  # 背景
    return mask_map


def sam_class_map(mask):
    # 255为未确定的
    mask_map = np.zeros_like(mask)
    mask_map[np.isin(mask, [255])] = 1  # 线粒体
    mask_map[np.isin(mask, [0])] = 255  # 未确定的
    return mask_map


def sparse_map(mask):
    # 255为未确定的
    mask_map = np.zeros_like(mask)
    mask_map[np.isin(mask, [255])] = 1  # 线粒体
    mask_map[np.isin(mask, [0])] = 0  # 未确定的
    mask_map[np.isin(mask, [128])] = 0  # 背景
    return mask_map


def label_class_map(mask):
    # 255为未确定的
    mask_map = np.zeros_like(mask)
    mask_map[np.isin(mask, [255])] = 1  # 线粒体
    mask_map[np.isin(mask, [128])] = 0  # 背景
    return mask_map


def instance_dilate(label, iter1=1, iter2=4, kernel_size=(4, 4)):
    kernel = np.ones(kernel_size, np.uint8)
    img_1 = cv2.dilate(label, kernel, iterations=iter1)
    img_2 = cv2.dilate(label, kernel, iterations=iter2)
    label[(img_2 == 255) & (img_1 != 255)] = 128
    return label


# --------------------------------------------------------------------------------
# Define indices for labelled, unlabelled training images, and test images
# --------------------------------------------------------------------------------


def get_k_idx(root, train=True):
    root = os.path.expanduser(root)
    if train:
        file_list = glob.glob(root + "/train/img/*.png")
    else:
        file_list = glob.glob(root + "/test/img/*.png")
    idx_list = [int(file[file.find("0") : file.rfind(".")]) for file in file_list]
    idx_list.sort()

    if train:
        return idx_list, idx_list
    else:
        return idx_list


def get_cvlab_idx(root, train=True):
    root = os.path.expanduser(root)
    if train:
        file_list = glob.glob(root + "/train/img/*.png")
    else:
        file_list = glob.glob(root + "/test/img/*.png")
    idx_list = [int(file[re.search(r"\d", file).span()[0] : file.rfind(".")]) for file in file_list]
    idx_list.sort()

    if train:
        return idx_list, idx_list
    else:
        return idx_list


def get_R_idx(root, train=True):
    root = os.path.expanduser(root)
    if train:
        # file_list = glob.glob(root + '/train/img_crop_10/*.png')
        file_list = glob.glob(root + "/train/img_crop/*.png")
        # idx_list = [int(file[re.search(r'train\d',file).span()[1]: file.rfind('.')]) for file in file_list]
        idx_list = [int(file[re.search(r"train\d", file).span()[1] - 1 : file.rfind(".")]) for file in file_list]
    else:
        file_list = glob.glob(root + "/test/img_4096/im*.png")
        idx_list = [int(file[re.search(r"im\d", file).span()[1] : file.rfind(".")]) for file in file_list]
    idx_list.sort()
    if train:
        return idx_list, idx_list
    else:
        return idx_list


def get_H_idx(root, train=True):
    root = os.path.expanduser(root)
    if train:
        # file_list = glob.glob(root + '/train/img_crop_10/*.png')
        file_list = glob.glob(root + "/train/img_crop/*.png")
        idx_list = [int(file[re.search(r"train\d", file).span()[1] - 1 : file.rfind(".")]) for file in file_list]
    else:
        file_list = glob.glob(root + "/test/img_4096/im*.png")
        idx_list = [int(file[re.search(r"im\d", file).span()[1] : file.rfind(".")]) for file in file_list]
    idx_list.sort()
    if train:
        return idx_list, idx_list
    else:
        return idx_list


def get_Stem_idx(root, train=True):
    root = os.path.expanduser(root)
    if train:
        file_list = glob.glob(root + "/train/img/*.png")
    else:
        file_list = glob.glob(root + "/valid/img/*.png")
    idx_list = [int(file[re.search(r"\d", file).span()[0] : file.rfind(".")]) for file in file_list]
    idx_list.sort()

    if train:
        return idx_list, idx_list
    else:
        return idx_list


# --------------------------------------------------------------------------------
# Create dataset in PyTorch format
# --------------------------------------------------------------------------------
class BuildDataset(Dataset):
    def __init__(
        self,
        root,
        dataset,
        idx_list,
        crop_size=(512, 512),
        scale_size=(0.5, 2.0),
        augmentation=True,
        train=True,
        apply_partial=None,
        partial_seed=None,
        DPO=False,
        heatmap_root=None,
        weak=None,
        target=False,
        H_dilate=False,
        kernal=(61,61)
    ):
        self.root = os.path.expanduser(root)
        self.train = train
        self.crop_size = crop_size
        self.augmentation = augmentation
        self.dataset = dataset
        self.idx_list = idx_list
        self.scale_size = scale_size
        self.apply_partial = apply_partial
        self.partial_seed = partial_seed
        self.DPO = DPO
        self.heatmap_root = heatmap_root
        self.weak = weak
        self.target = target
        self.H_dilate = H_dilate
        self.kernal = kernal

    def __getitem__(self, index):

        if self.dataset == "cvlab":
            if self.DPO:
                image_root = Image.open(self.root + "/train/img/train{}.png".format(str(self.idx_list[index]).zfill(3)))
                if self.apply_partial == "None":
                    label_root = Image.open(self.root + "/train/lab/train{}.png".format(str(self.idx_list[index]).zfill(3)))
                    label_root = Image.fromarray(label_class_map(np.array(label_root)))
                    image_root = Image.fromarray(np.stack((image_root, image_root, image_root), axis=-1))
                    point_map = point_root = img_generate_centerpoint(label_root)
                    heatmap, _ = pointmap_generate_gaussian(point_root, self.kernal)
                elif self.apply_partial == "0%":
                    label_root = Image.open(
                        self.root + "/train/pre_{}_DPO/train{}.png".format(self.apply_partial, str(self.idx_list[index]).zfill(3))
                    )
                    label_root = Image.fromarray(label_class_map(np.array(label_root)))
                    image_root = Image.fromarray(np.stack((image_root, image_root, image_root), axis=-1))
                    point_root = Image.fromarray(np.zeros_like((np.array(label_root))))
                    point_map = point_root
                    heatmap, _ = pointmap_generate_gaussian(point_root, self.kernal)
                else:
                    label_root = Image.open(
                        self.root + "/train/pre_{}_DPO/train{}.png".format(self.apply_partial, str(self.idx_list[index]).zfill(3))
                    )

                    image_root = Image.fromarray(np.stack((image_root, image_root, image_root), axis=-1))
                    label_root = Image.fromarray(label_class_map(np.array(label_root)))
                    point_map = point_root = img_generate_centerpoint(label_root)
                    heatmap, _ = pointmap_generate_gaussian(point_root, self.kernal)
                image, label, heatmap, point_map = transform_DPO(
                    image_root,
                    label_root,
                    heatmap,
                    point_map,
                    crop_size=self.crop_size,
                    scale_size=self.scale_size,
                    augmentation=self.augmentation,
                    weak=self.weak,
                )
                promt_point, point_map, instance_label, pointmap_partial = heatmap_label2prompt_heatmap(point_map, label, drop=True)
                return image, label.squeeze(0), point_map, heatmap, promt_point, instance_label.squeeze(0), pointmap_partial.squeeze(0)
            elif self.train:
                image_root = Image.open(self.root + "/train/img/train{}.png".format(str(self.idx_list[index]).zfill(3)))
                if self.apply_partial == "None":
                    label_root = Image.open(self.root + "/train/lab/train{}.png".format(str(self.idx_list[index]).zfill(3)))
                    image_root = Image.fromarray(np.stack((image_root, image_root, image_root), axis=-1))
                    if self.target:
                        label_root = Image.fromarray(np.ones_like((np.array(label_root))) * 255)
                    else:
                        label_root = Image.fromarray(label_class_map(np.array(label_root)))
                    point_root = Image.fromarray(np.zeros_like((np.array(label_root))))
                elif self.apply_partial == "0%":
                    label_root = Image.open(self.root + "/train/point_{}_UDA/train{}.png".format(self.apply_partial, str(self.idx_list[index]).zfill(3)))
                    image_root = Image.fromarray(np.stack((image_root, image_root, image_root), axis=-1))
                    if self.target:
                        label_root = Image.fromarray(np.ones_like((np.array(label_root))) * 255)
                    else:
                        label_root = Image.fromarray(label_class_map(np.array(label_root)))
                    point_root = Image.fromarray(np.zeros_like((np.array(label_root))))
                else:
                    label_root = Image.open(
                        self.root + "/train/point_{}/train{}.png".format(self.apply_partial, str(self.idx_list[index]).zfill(3))
                    )
                    image_root = Image.fromarray(np.stack((image_root, image_root, image_root), axis=-1))
                    label_root = Image.fromarray(label_class_map(np.array(label_root)))
                    point_root = img_generate_centerpoint(label_root)
                point_map = point_root
                heatmap, _ = pointmap_generate_gaussian(point_root, self.kernal)
                image, label, heatmap, point_map = transform(
                    image_root,
                    label_root,
                    heatmap,
                    point_map,
                    crop_size=self.crop_size,
                    scale_size=self.scale_size,
                    augmentation=self.augmentation,
                    weak=self.weak,
                )
                promt_point, point_map, instance_label, pointmap_partial = heatmap_label2prompt_heatmap(point_map, label, drop=True)
                return image, label.squeeze(0), point_map, heatmap, promt_point, instance_label.squeeze(0), pointmap_partial.squeeze(0)

            else:
                image_root = Image.open(self.root + "/test/img/test{}.png".format(str(self.idx_list[index]).zfill(3)))
                label_root = Image.open(self.root + "/test/lab/test{}.png".format(str(self.idx_list[index]).zfill(3)))
                image_root = Image.fromarray(np.stack((image_root, image_root, image_root), axis=-1))
                label_root = Image.fromarray(label_class_map(np.array(label_root)))
                point_root = Image.open(self.root + "/test/point/test{}.png".format(str(self.idx_list[index]).zfill(3)))
                heatmap_root = point_root
            image, label, heatmap = transform(
                image=image_root,
                label=label_root,
                logits=None,
                point=heatmap_root,
                crop_size=self.crop_size,
                scale_size=self.scale_size,
                augmentation=self.augmentation,
                weak=self.weak,
            )
            return image, label.squeeze(0), heatmap

        if self.dataset == "R":
            if self.DPO:
                image_root = Image.open(self.root + "/train/img_crop/train{}.png".format(str(self.idx_list[index]).zfill(4)))
                if self.apply_partial == "None":
                    label_root = Image.open(self.root + "/train/lab_crop/train{}.png".format(str(self.idx_list[index]).zfill(4)))
                    image_root = Image.fromarray(np.stack((image_root, image_root, image_root), axis=-1))
                    label_root = Image.fromarray(label_class_map(np.array(label_root)))
                    point_root = Image.open(self.root + "/train/point_crop/train{}.png".format(str(self.idx_list[index]).zfill(4)))
                    point_map = point_root
                    heatmap, _ = pointmap_generate_gaussian(point_root, self.kernal)
                elif self.apply_partial == "0%":
                    label_root = Image.open(self.root + "/train/pre_crop_{}_DPO/train{}.png".format(self.apply_partial, str(self.idx_list[index]).zfill(4)))
                    image_root = Image.fromarray(np.stack((image_root, image_root, image_root), axis=-1))
                    label_root = Image.fromarray(label_class_map(np.array(label_root)))
                    point_map = point_root = Image.fromarray(np.zeros_like((np.array(label_root))))
                    heatmap, _ = pointmap_generate_gaussian(point_root, self.kernal)
                else:
                    label_root = Image.open(self.root + "/train/pre_crop_{}_DPO/train{}.png".format(self.apply_partial, str(self.idx_list[index]).zfill(4)))
                    image_root = Image.fromarray(np.stack((image_root, image_root, image_root), axis=-1))
                    label_root = Image.fromarray(label_class_map(np.array(label_root)))
                    point_root = Image.open(
                        self.root + "/train/point_crop_{}/train{}.png".format(self.apply_partial, str(self.idx_list[index]).zfill(4))
                    )
                    point_map = point_root
                    heatmap, _ = pointmap_generate_gaussian(point_root, self.kernal)
                image, label, heatmap, point_map = transform_DPO(
                    image_root,
                    label_root,
                    heatmap,
                    point_map,
                    crop_size=self.crop_size,
                    scale_size=self.scale_size,
                    augmentation=self.augmentation,
                    weak=self.weak,
                )
                promt_point, point_map, instance_label, pointmap_partial = heatmap_label2prompt_heatmap_Focus(point_map, label, drop=True)
                return image, label.squeeze(0), point_map, heatmap, promt_point, instance_label.squeeze(0), pointmap_partial.squeeze(0)
            elif self.train:
                image_root = Image.open(self.root + "/train/img_crop/train{}.png".format(str(self.idx_list[index]).zfill(4)))
                # image_root = Image.open(self.root + '/train/img_crop_10/train{}.png'.format(str(self.idx_list[index]).zfill(4)))
                if self.apply_partial == "None":
                    label_root = Image.open(self.root + "/train/lab_crop/train{}.png".format(str(self.idx_list[index]).zfill(4)))
                    # label_root = Image.open(self.root + '/train/lab_crop_10/train{}.png'.format(str(self.idx_list[index]).zfill(4)))
                    image_root = Image.fromarray(np.stack((image_root, image_root, image_root), axis=-1))
                    if self.target:
                        label_root = Image.fromarray(np.ones_like((np.array(label_root))) * 255)
                    else:
                        label_root = Image.fromarray(label_class_map(np.array(label_root)))
                    point_root = Image.open(self.root + "/train/point_crop/train{}.png".format(str(self.idx_list[index]).zfill(4)))
                    # point_root = img_generate_centerpoint(label_root)
                elif self.apply_partial == "0%":
                    label_root = Image.open(self.root + "/train/point_crop_{}_UDA/train{}.png".format(self.apply_partial, str(self.idx_list[index]).zfill(4)))
                    # label_root = Image.open(self.root + '/train/lab_crop_10/train{}.png'.format(str(self.idx_list[index]).zfill(4)))
                    image_root = Image.fromarray(np.stack((image_root, image_root, image_root), axis=-1))
                    label_root = Image.fromarray(label_class_map(np.array(label_root)))
                    point_root = label_root
                    # point_root = img_generate_centerpoint(label_root)
                else:
                    # label_root = Image.open(
                    #     self.root + "/train/lab_crop_{}/train{}.png".format(self.apply_partial, str(self.idx_list[index]).zfill(4))
                    # )
                    label_root = Image.open(self.root + "/train/point_crop_{}/train{}.png".format(self.apply_partial, str(self.idx_list[index]).zfill(4)))
                    image_root = Image.fromarray(np.stack((image_root, image_root, image_root), axis=-1))
                    label_root = Image.fromarray(label_class_map(np.array(label_root)))
                    # heatmap_root = Image.fromarray(np.load('heatmap_label/R/1%_2/train' + '/{}.npy'.format(str(self.idx_list[index])))[0])
                    # heatmap_root = img_generate_centerpoint(Image.fromarray(label_class_map(np.array(Image.open(self.root + '/train/lab_crop_10/train{}.png'.format(str(self.idx_list[index]).zfill(4)))))))
                    # point_root = Image.open(self.root + "/train/point_crop_{}/train{}.png".format(self.apply_partial, str(self.idx_list[index]).zfill(4)))
                    point_root = label_root
                point_map = point_root
                heatmap, _ = pointmap_generate_gaussian(point_root, self.kernal)
                image, label, heatmap, point_map = transform(
                    image_root,
                    label_root,
                    heatmap,
                    point_map,
                    crop_size=self.crop_size,
                    scale_size=self.scale_size,
                    augmentation=self.augmentation,
                    weak=self.weak,
                )
                promt_point, point_map, instance_label, pointmap_partial = heatmap_label2prompt_heatmap(point_map, label, drop=True)
                # promt_point, heatmap = get_point_prompt_heatmap(heatmap, drop=True)
                return image, label.squeeze(0), point_map, heatmap, promt_point, instance_label.squeeze(0), pointmap_partial.squeeze(0)

            else:
                image_root = Image.open(self.root + "/test/img_4096/im{}.png".format(str(self.idx_list[index]).zfill(4)))
                label_root = Image.open(self.root + "/test/lab_4096/im{}.png".format(str(self.idx_list[index]).zfill(4)))
                image_root = Image.fromarray(np.stack((image_root, image_root, image_root), axis=-1))
                label_root = Image.fromarray(label_class_map(np.array(label_root)))
                # heatmap_root = Image.fromarray(np.load('heatmap_label/R/1%_2/test' + '/{}.npy'.format(str(self.idx_list[index])))[0])
                point_root = Image.open(self.root + "/test/point_4096/im{}.png".format(str(self.idx_list[index]).zfill(4)))
                heatmap_root = point_root
            image, label, heatmap = transform(
                image=image_root,
                label=label_root,
                logits=None,
                point=heatmap_root,
                crop_size=self.crop_size,
                scale_size=self.scale_size,
                augmentation=self.augmentation,
                weak=self.weak,
            )
            return image, label.squeeze(0), heatmap

        if self.dataset == "H":
            if self.DPO:
                image_root = Image.open(self.root + "/train/img_crop/train{}.png".format(str(self.idx_list[index]).zfill(4)))
                if self.apply_partial == "None":
                    label_root = Image.open(self.root + "/train/lab_crop/train{}.png".format(str(self.idx_list[index]).zfill(4)))
                    label_root = Image.fromarray(label_class_map(np.array(label_root)))
                    image_root = Image.fromarray(np.stack((image_root, image_root, image_root), axis=-1))
                    point_root = Image.open(self.root + "/train/point_crop/train{}.png".format(str(self.idx_list[index]).zfill(4)))
                elif self.apply_partial == "0%":
                    label_root = Image.open(
                        self.root + "/train/pre_crop_{}_DPO/train{}.png".format(self.apply_partial, str(self.idx_list[index]).zfill(4))
                    )
                    label_root = Image.fromarray(label_class_map(np.array(label_root)))
                    image_root = Image.fromarray(np.stack((image_root, image_root, image_root), axis=-1))
                    point_root = Image.fromarray(np.zeros_like((np.array(label_root))))
                else:
                    label_root = Image.open(self.root + "/train/pre_crop_{}_DPO/train{}.png".format(self.apply_partial, str(self.idx_list[index]).zfill(4)))

                    image_root = Image.fromarray(np.stack((image_root, image_root, image_root), axis=-1))
                    label_root = Image.fromarray(label_class_map(np.array(label_root)))
                    point_root = Image.open(
                        self.root + "/train/point_crop_{}/train{}.png".format(self.apply_partial, str(self.idx_list[index]).zfill(4))
                    )
                    # point_root = Image.open(
                    #     self.root + "/train/lab_crop/train{}.png".format(str(self.idx_list[index]).zfill(4))
                    # )
                point_map = point_root
                heatmap, _ = pointmap_generate_gaussian(point_root, self.kernal)

                image, label, heatmap, point_map = transform_DPO(
                    image=image_root,
                    label=label_root,
                    logits=heatmap,
                    point=point_map,
                    crop_size=self.crop_size,
                    scale_size=self.scale_size,
                    augmentation=self.augmentation,
                    weak=self.weak,
                )
                promt_point, point_map, instance_label, pointmap_partial = heatmap_label2prompt_heatmap_Focus(point_map, label, drop=True)
                return image, label.squeeze(0), point_map, heatmap, promt_point, instance_label.squeeze(0), pointmap_partial.squeeze(0)
                # return image, label.squeeze(0), point_map, point_map, point_map, point_map, point_map

            elif self.train:
                image_root = Image.open(self.root + "/train/img_crop/train{}.png".format(str(self.idx_list[index]).zfill(4)))
                if self.apply_partial == "None":
                    if(self.H_dilate):
                    #only using H gac when H to cvlab
                        label_root = Image.open(self.root + "/train/lab_gac_crop/train{}.png".format(str(self.idx_list[index]).zfill(4)))
                    else:
                        label_root = Image.open(self.root + '/train/lab_crop/train{}.png'.format(str(self.idx_list[index]).zfill(4)))

                    image_root = Image.fromarray(np.stack((image_root, image_root, image_root), axis=-1))
                    if self.target:
                        label_root = Image.fromarray(np.ones_like((np.array(label_root))) * 255)
                    else:
                        label_root = Image.fromarray(label_class_map(np.array(label_root)))
                    # point_root = Image.open(
                    #     self.root + "/train/point_crop/train{}.png".format(str(self.idx_list[index]).zfill(4)))
                    point_root = img_generate_centerpoint(label_root)
                elif self.apply_partial == "0%":
                    image_root = Image.fromarray(np.stack((image_root, image_root, image_root), axis=-1))
                    label_root = Image.open(self.root + "/train/point_crop_{}_UDA/train{}.png".format(self.apply_partial, str(self.idx_list[index]).zfill(4)))
                    label_root = Image.fromarray(label_class_map(np.array(label_root)))
                    point_root = label_root
                else:
                    # label_root = Image.open(
                    #     self.root + "/train/lab_crop_{}/train{}.png".format(self.apply_partial, str(self.idx_list[index]).zfill(4))
                    # )
                    label_root = Image.open(
                        self.root + "/train/point_crop_{}/train{}.png".format(self.apply_partial, str(self.idx_list[index]).zfill(4))
                    )
                    image_root = Image.fromarray(np.stack((image_root, image_root, image_root), axis=-1))
                    label_root = Image.fromarray(label_class_map(np.array(label_root)))
                    # heatmap_root = Image.fromarray(np.load('heatmap_label/H/1%_2/train' + '/{}.npy'.format(str(self.idx_list[index])))[0])
                    # point_root = Image.open(
                    #     self.root + "/train/point_crop_{}/train{}.png".format(self.apply_partial, str(self.idx_list[index]).zfill(4))
                    # )
                    point_root = img_generate_centerpoint(label_root)
                point_map = point_root
                heatmap, _ = pointmap_generate_gaussian(point_root, self.kernal)
                image, label, heatmap, point_map = transform(
                    image=image_root,
                    label=label_root,
                    logits=heatmap,
                    point=point_map,
                    crop_size=self.crop_size,
                    scale_size=self.scale_size,
                    augmentation=self.augmentation,
                    weak=self.weak,
                )
                promt_point, point_map, instance_label, pointmap_partial = heatmap_label2prompt_heatmap(point_map, label, drop=True)
                # promt_point, heatmap = get_point_prompt_heatmap(heatmap, drop=True)
                return image, label.squeeze(0), point_map, heatmap, promt_point, instance_label.squeeze(0), pointmap_partial.squeeze(0)
                # return image, label.squeeze(0), heatmap, promt_point
            else:
                image_root = Image.open(self.root + "/test/img_4096/im{}.png".format(str(self.idx_list[index]).zfill(4)))
                label_root = Image.open(self.root + "/test/lab_4096/im{}.png".format(str(self.idx_list[index]).zfill(4)))
                image_root = Image.fromarray(np.stack((image_root, image_root, image_root), axis=-1))
                label_root = Image.fromarray(label_class_map(np.array(label_root)))
                # heatmap_root = Image.fromarray(np.load('heatmap_label/H/1%_2/test' + '/{}.npy'.format(str(self.idx_list[index])))[0])
                # heatmap_root, scale = img_generate_gaussian(label_root, (61,61))
                point_root = Image.open(self.root + "/test/point_4096/im{}.png".format(str(self.idx_list[index]).zfill(4)))
                heatmap_root = point_root
            image, label, heatmap = transform(
                image=image_root,
                label=label_root,
                logits=None,
                point=heatmap_root,
                crop_size=self.crop_size,
                scale_size=self.scale_size,
                augmentation=self.augmentation,
                weak=self.weak,
            )
            return image, label.squeeze(0), heatmap

        if self.dataset == "Stem":
            if self.DPO:
                image_root = Image.open(self.root + "/train/img/train{}.png".format(str(self.idx_list[index]).zfill(3)))
                if self.apply_partial == "None":
                    label_root = Image.open(self.root + "/train/lab/train{}.png".format(str(self.idx_list[index]).zfill(3)))
                    label_root = Image.fromarray(label_class_map(np.array(label_root)))
                    image_root = Image.fromarray(np.stack((image_root, image_root, image_root), axis=-1))
                    point_map = point_root = img_generate_centerpoint(label_root)
                    heatmap, _ = pointmap_generate_gaussian(point_root, self.kernal)
                elif self.apply_partial == "0%":
                    label_root = Image.open(
                        self.root + "/train/pre_{}_DPO/train{}.png".format(self.apply_partial, str(self.idx_list[index]).zfill(3))
                    )
                    label_root = Image.fromarray(label_class_map(np.array(label_root)))
                    image_root = Image.fromarray(np.stack((image_root, image_root, image_root), axis=-1))
                    point_root = Image.fromarray(np.zeros_like((np.array(label_root))))
                    point_map = point_root
                    heatmap, _ = pointmap_generate_gaussian(point_root, self.kernal)
                else:
                    label_root = Image.open(
                        self.root + "/train/pre_{}_DPO/train{}.png".format(self.apply_partial, str(self.idx_list[index]).zfill(3))
                    )

                    image_root = Image.fromarray(np.stack((image_root, image_root, image_root), axis=-1))
                    label_root = Image.fromarray(label_class_map(np.array(label_root)))
                    point_map = point_root = img_generate_centerpoint(label_root)
                    heatmap, _ = pointmap_generate_gaussian(point_root, self.kernal)
                image, label, heatmap, point_map = transform_DPO(
                    image_root,
                    label_root,
                    heatmap,
                    point_map,
                    crop_size=self.crop_size,
                    scale_size=self.scale_size,
                    augmentation=self.augmentation,
                    weak=self.weak,
                )
                promt_point, point_map, instance_label, pointmap_partial = heatmap_label2prompt_heatmap(point_map, label, drop=True)
                return image, label.squeeze(0), point_map, heatmap, promt_point, instance_label.squeeze(0), pointmap_partial.squeeze(0)
            elif self.train:
                image_root = Image.open(self.root + "/train/img/train{}.png".format(str(self.idx_list[index]).zfill(3)))
                if self.apply_partial == None:
                    label_root = Image.open(self.root + "/train/lab/train{}.png".format(str(self.idx_list[index]).zfill(3)))
                    image_root = Image.fromarray(np.stack((image_root, image_root, image_root), axis=-1))
                    label_root = Image.fromarray(label_class_map(np.array(label_root)))
                    point_root = Image.fromarray(np.zeros_like((np.array(label_root))))
                elif self.apply_partial == "0%":
                    label_root = Image.open(self.root + "/train/point_{}_UDA/train{}.png".format(self.apply_partial, str(self.idx_list[index]).zfill(3)))
                    image_root = Image.fromarray(np.stack((image_root, image_root, image_root), axis=-1))
                    label_root = Image.fromarray(label_class_map(np.array(label_root)))
                    point_root = label_root
                else:
                    label_root = Image.open(
                        self.root + "/train/point_{}/train{}.png".format(self.apply_partial, str(self.idx_list[index]).zfill(3))
                    )
                    image_root = Image.fromarray(np.stack((image_root, image_root, image_root), axis=-1))
                    label_root = Image.fromarray(label_class_map(np.array(label_root)))
                    point_root = img_generate_centerpoint(label_root)
                point_map = point_root
                heatmap, _ = pointmap_generate_gaussian(point_root, self.kernal)
                image, label, heatmap, point_map = transform(
                    image_root,
                    label_root,
                    heatmap,
                    point_map,
                    crop_size=self.crop_size,
                    scale_size=self.scale_size,
                    augmentation=self.augmentation,
                    weak=self.weak,
                )
                promt_point, point_map, instance_label, pointmap_partial = heatmap_label2prompt_heatmap(point_map, label, drop=True)
                return image, label.squeeze(0), point_map, heatmap, promt_point, instance_label.squeeze(0), pointmap_partial.squeeze(0)

            else:
                image_root = Image.open(self.root + "/valid/img/valid{}.png".format(str(self.idx_list[index]).zfill(3)))
                label_root = Image.open(self.root + "/valid/lab/valid{}.png".format(str(self.idx_list[index]).zfill(3)))
                image_root = Image.fromarray(np.stack((image_root, image_root, image_root), axis=-1))
                label_root = Image.fromarray(label_class_map(np.array(label_root)))
                point_root = Image.open(self.root + "/valid/point/valid{}.png".format(str(self.idx_list[index]).zfill(3)))
                heatmap_root = point_root
            image, label, heatmap = transform(
                image=image_root,
                label=label_root,
                logits=None,
                point=heatmap_root,
                crop_size=self.crop_size,
                scale_size=self.scale_size,
                augmentation=self.augmentation,
                weak=self.weak,
            )
            return image, label.squeeze(0), heatmap
        
    def __len__(self):
        return len(self.idx_list)

# --------------------------------------------------------------------------------
# Create data loader in PyTorch format
# --------------------------------------------------------------------------------
class BuildDataLoader:
    def __init__(self, dataset, crop_size, batch_size, target=False):
        self.dataset = dataset

        if dataset == "K++":
            self.data_path = "dataset/K++"
            self.im_size = [1463, 1613]
            self.test_size = [1334, 1553]
            self.crop_size = [512, 512]
            self.num_segments = 2
            self.scale_size = (1.0, 1.0)
            self.batch_size = batch_size
            self.train_l_idx, self.train_u_idx = get_k_idx(self.data_path, train=True)
            self.test_idx = get_k_idx(self.data_path, train=False)
        if dataset == "cvlab":
            self.data_path = "dataset/cvlab"
            self.im_size = [768, 1024]
            self.test_size = [768, 1024]
            self.crop_size = [crop_size, crop_size]
            self.num_segments = 2
            self.scale_size = (1.0, 1.0)
            self.batch_size = batch_size
            self.train_l_idx, self.train_u_idx = get_cvlab_idx(self.data_path, train=True)
            self.test_idx = get_cvlab_idx(self.data_path, train=False)
            self.target = target
        if dataset == "R":
            self.data_path = "dataset/R"
            self.im_size = [1024, 1024]
            self.test_size = [4096, 4096]
            self.crop_size = [crop_size, crop_size]
            self.num_segments = 2
            self.scale_size = (1.0, 1.0)
            self.batch_size = batch_size
            self.train_l_idx, self.train_u_idx = get_R_idx(self.data_path, train=True)
            self.test_idx = get_R_idx(self.data_path, train=False)
            self.target = target
        if dataset == "H":
            self.data_path = "dataset/H"
            self.im_size = [1024, 1024]
            self.test_size = [4096, 4096]
            self.crop_size = [crop_size, crop_size]
            self.num_segments = 2
            self.scale_size = (1.0, 1.0)
            self.batch_size = batch_size
            self.train_l_idx, self.train_u_idx = get_H_idx(self.data_path, train=True)
            self.test_idx = get_H_idx(self.data_path, train=False)
            self.target = target
        if dataset == "Stem":
            self.data_path = "dataset/Stem"
            self.im_size = [1000, 1000]
            self.test_size = [1000, 1000]
            self.crop_size = [crop_size, crop_size]
            self.num_segments = 2
            self.scale_size = (1.0, 1.0)
            self.batch_size = batch_size
            self.train_l_idx, self.train_u_idx = get_Stem_idx(self.data_path, train=True)
            self.test_idx = get_Stem_idx(self.data_path, train=False)
            self.target = target
            
    def build(self, partial=None, partial_seed=None, weak=None, num_samples=None,augment=False,H_dilate=False,kernal=(61,61)):
        train_l_dataset = BuildDataset(
            self.data_path,
            self.dataset,
            self.train_l_idx,
            crop_size=self.crop_size,
            scale_size=self.scale_size,
            augmentation=augment,
            train=True,
            apply_partial=partial,
            partial_seed=partial_seed,
            weak=weak,
            target=self.target,
            H_dilate = H_dilate,
            kernal = kernal
        )

        test_dataset = BuildDataset(
            self.data_path,
            self.dataset,
            self.test_idx,
            crop_size=self.test_size,
            scale_size=(1.0, 1.0),
            augmentation=False,
            train=False,
        )

        if num_samples == None:
            num_samples = len(train_l_dataset)
        train_l_loader = torch.utils.data.DataLoader(
            train_l_dataset,
            batch_size=self.batch_size,
            sampler=sampler.RandomSampler(data_source=train_l_dataset, replacement=False, num_samples=num_samples),
            drop_last=True,
            pin_memory=True,
            num_workers=8,
        )

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)

        return train_l_loader, test_loader

    def build_det(self, partial=None, partial_seed=None, weak=None, DPO=False, H_dilate=False,augment=False,kernal=(61,61)):
        train_l_dataset = BuildDataset(
            self.data_path,
            self.dataset,
            self.train_l_idx,
            crop_size=self.crop_size,
            scale_size=self.scale_size,
            augmentation=augment,
            train=True,
            apply_partial=partial,
            partial_seed=partial_seed,
            target=self.target,
            DPO=DPO,
            H_dilate=H_dilate,
            kernal=kernal
        )

        test_dataset = BuildDataset(
            self.data_path,
            self.dataset,
            self.test_idx,
            crop_size=self.test_size,
            scale_size=(1.0, 1.0),
            augmentation=False,
            train=False,
        )

        train_l_loader = torch.utils.data.DataLoader(
            train_l_dataset,
            batch_size=self.batch_size,
            sampler=sampler.RandomSampler(data_source=train_l_dataset, replacement=False, num_samples=len(train_l_dataset)),
            drop_last=True,
            pin_memory=True,
            num_workers=8,
        )

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=False, pin_memory=True, num_workers=8)

        return train_l_loader, test_loader
