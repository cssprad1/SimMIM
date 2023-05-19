# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhenda Xie
# --------------------------------------------------------

import math
import os
import random
import numpy as np

from skimage.transform import resize

import torch
import torch.distributed as dist
import torchvision.transforms as T
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data._utils.collate import default_collate
# from torchvision.datasets import ImageFolder
# from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class MODISDataset(Dataset):

    IMAGE_PATH = os.path.join("images")

    def __init__(
        self,
        config,
        data_paths: list,
        split: str,
        img_size: tuple = (192, 192),
        transform=None,
    ):
        self.config = config
        self.img_size = img_size
        self.transform = transform
        self.split = split
        self.data_paths = data_paths
        # self.img_path = os.path.join(self.data_path, self.IMAGE_PATH)
        # self.img_list = self.get_filenames(self.img_path)

        self.img_list = []
        for data_path in data_paths:
            img_path = os.path.join(data_path, self.IMAGE_PATH)
            self.img_list.extend(self.get_filenames(img_path))
        n_items = len(self.img_list)
        print(f'> Found {n_items} patches for this dataset ({split})')
        
        if config.MODEL.TYPE in ['swin', 'swinv2']:
            model_patch_size = config.MODEL.SWIN.PATCH_SIZE
        elif config.MODEL.TYPE == 'vit':
            model_patch_size = config.MODEL.VIT.PATCH_SIZE
        else:
            raise NotImplementedError

        self.mask_generator = MaskGenerator(
            input_size=config.DATA.IMG_SIZE,
            mask_patch_size=config.DATA.MASK_PATCH_SIZE,
            model_patch_size=model_patch_size,
            mask_ratio=config.DATA.MASK_RATIO,
        )

        # Split between train and valid set (80/20)
        # random_inst = random.Random(12345)  # for repeatability
        # n_items = len(self.img_list)
        # idxs = random_inst.sample(range(n_items), n_items // 5)
        # if self.split == "train":
        #    idxs = [idx for idx in range(n_items) if idx not in idxs]
        # self.img_list = [self.img_list[i] for i in idxs]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx, transpose=True):

        # load image
        img = np.load(self.img_list[idx])

        img = np.clip(img, 0, 1.0)
        img = resize(img, self.img_size + (img.shape[-1],))

        # perform transformations
        img = self.transform(img)
        mask = self.mask_generator()

        return img, mask

    def get_filenames(self, path):
        """
        Returns a list of absolute paths to images inside given `path`
        """
        files_list = []
        for filename in os.listdir(path):
            files_list.append(os.path.join(path, filename))
        return files_list


class RandomResizedCropNP(object):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0/4.0, 4.0/3.0)):
        self.size = size
        self.scale = scale
        self.ratio = ratio

    def __call__(self, img):
        height, width = img.shape[:2]
        area = height * width
        
        for attempt in range(10):
            target_area = np.random.uniform(*self.scale) * area
            aspect_ratio = np.random.uniform(*self.ratio)
            
            w = int(round(np.sqrt(target_area * aspect_ratio)))
            h = int(round(np.sqrt(target_area / aspect_ratio)))
            
            if np.random.random() < 0.5:
                w, h = h, w
            
            if w <= width and h <= height:
                x1 = np.random.randint(0, width - w + 1)
                y1 = np.random.randint(0, height - h + 1)
                cropped = img[y1:y1+h, x1:x1+w, :]
                cropped = np.moveaxis(cropped, -1, 0)
                cropped_resized = torch.nn.functional.interpolate(torch.from_numpy(cropped).unsqueeze(0), size=self.size, mode='bicubic', align_corners=False)
                cropped_squeezed_numpy = cropped_resized.squeeze().numpy()
                cropped_squeezed_numpy = np.moveaxis(cropped_squeezed_numpy, 0, -1)
                return cropped_squeezed_numpy
        
        # if crop was not successful after 10 attempts, use center crop
        w = min(width, height)
        x1 = (width - w) // 2
        y1 = (height - w) // 2
        cropped = img[y1:y1+w, x1:x1+w, :]
        cropped = np.moveaxis(cropped, -1, 0)
        cropped_resized = torch.nn.functional.interpolate(torch.from_numpy(cropped).unsqueeze(0), size=self.size, mode='bicubic', align_corners=False)
        cropped_squeezed_numpy = cropped_resized.squeeze().numpy()
        cropped_squeezed_numpy = np.moveaxis(cropped_squeezed_numpy, 0, -1)
        return cropped_squeezed_numpy


class MaskGenerator:
    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1

        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)

        return mask


class SimMIMTransform:
    def __init__(self, config):
        self.transform_img = T.Compose([
            RandomResizedCropNP(config.DATA.IMG_SIZE, scale=(0.67, 1.), ratio=(3. / 4., 4. / 3.)),
            T.ToTensor(),
        ])

        if config.MODEL.TYPE in ['swin', 'swinv2']:
            model_patch_size = config.MODEL.SWIN.PATCH_SIZE
        elif config.MODEL.TYPE == 'vit':
            model_patch_size = config.MODEL.VIT.PATCH_SIZE
        else:
            raise NotImplementedError

        self.mask_generator = MaskGenerator(
            input_size=config.DATA.IMG_SIZE,
            mask_patch_size=config.DATA.MASK_PATCH_SIZE,
            model_patch_size=model_patch_size,
            mask_ratio=config.DATA.MASK_RATIO,
        )

    def __call__(self, img):
        img = self.transform_img(img)
        mask = self.mask_generator()

        return img, mask


class SimMIMTransformWNorm:
    def __init__(self, config):
        self.transform_img = T.Compose([
            RandomResizedCropNP(config.DATA.IMG_SIZE, scale=(0.67, 1.), ratio=(3. / 4., 4. / 3.)),
            T.ToTensor(),
            T.Normalize(
                    mean=[0.0173, 0.0332, 0.0088,
                          0.0136, 0.0381, 0.0348, 0.0249],
                    std=[0.0150, 0.0127, 0.0124,
                         0.0128, 0.0120, 0.0159, 0.0164]
                ),
        ])

        if config.MODEL.TYPE in ['swin', 'swinv2']:
            model_patch_size = config.MODEL.SWIN.PATCH_SIZE
        elif config.MODEL.TYPE == 'vit':
            model_patch_size = config.MODEL.VIT.PATCH_SIZE
        else:
            raise NotImplementedError

        self.mask_generator = MaskGenerator(
            input_size=config.DATA.IMG_SIZE,
            mask_patch_size=config.DATA.MASK_PATCH_SIZE,
            model_patch_size=model_patch_size,
            mask_ratio=config.DATA.MASK_RATIO,
        )

    def __call__(self, img):
        img = self.transform_img(img)
        mask = self.mask_generator()

        return img, mask



def collate_fn(batch):
    if not isinstance(batch[0][0], tuple):
        return default_collate(batch)
    else:
        batch_num = len(batch)
        ret = []
        for item_idx in range(len(batch[0][0])):
            if batch[0][0][item_idx] is None:
                ret.append(None)
            else:
                ret.append(default_collate(
                    [batch[i][0][item_idx] for i in range(batch_num)]))
        ret.append(default_collate([batch[i][1] for i in range(batch_num)]))
        return ret


def build_loader_simmim(config, logger):
    transform = SimMIMTransformWNorm(config)
    logger.info(f'Pre-train data transform:\n{transform}')

    dataset = MODISDataset(config,
                           config.DATA.DATA_PATH,
                           split="train",
                           img_size=(192, 192),
                           transform=transform)
    # dataset = ImageFolder(config.DATA.DATA_PATH, transform)
    logger.info(f'Build dataset: train images = {len(dataset)}')

    sampler = DistributedSampler(
        dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True)
    dataloader = DataLoader(dataset, config.DATA.BATCH_SIZE, sampler=sampler,
                            num_workers=config.DATA.NUM_WORKERS, pin_memory=True, drop_last=True, collate_fn=collate_fn)

    return dataloader
