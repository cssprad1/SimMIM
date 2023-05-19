# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhenda Xie
# --------------------------------------------------------

import os
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, DistributedSampler
import torchvision.transforms as T
from torchvision import datasets, transforms
# from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
# from timm.data import create_transform
# from timm.data.transforms import _pil_interp
import numpy as np
from skimage.transform import resize
import random


class MODISDataset(Dataset):

    IMAGE_PATH = os.path.join("images")
    MASK_PATH = os.path.join("labels")

    def __init__(
        self,
        data_path: str,
        split: str,
        img_size: tuple = (256, 256),
        transform=None,
    ):
        self.img_size = img_size
        self.transform = transform
        self.split = split
        self.data_path = data_path
        self.img_path = os.path.join(self.data_path, self.IMAGE_PATH)
        self.mask_path = os.path.join(self.data_path, self.MASK_PATH)
        self.img_list = self.get_filenames(self.img_path)
        self.mask_list = self.get_filenames(self.mask_path)

        # Split between train and valid set (80/20)
        random_inst = random.Random(12345)  # for repeatability
        n_items = len(self.img_list)
        idxs = set(random_inst.sample(range(n_items), n_items // 5))
        total_idxs = set(range(n_items))
        if self.split == "train":
            idxs = total_idxs - idxs
        self.img_list = [self.img_list[i] for i in idxs]
        self.mask_list = [self.mask_list[i] for i in idxs]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx, transpose=True):

        # load image
        img = np.load(self.img_list[idx])

        img = np.clip(img, 0, 1.0)
        img = resize(img, self.img_size + (img.shape[-1],))

        # load mask
        mask = np.load(self.mask_list[idx])
        if len(mask.shape) > 2:
            mask = resize(mask, self.img_size + (mask.shape[-1],))
            mask = np.argmax(mask, axis=-1)
        else:
            mask = resize(mask, self.img_size)

        # perform transformations
        if self.transform is not None:
            img = self.transform(img)

        return img, mask

    def get_filenames(self, path):
        """
        Returns a list of absolute paths to images inside given `path`
        """
        files_list = []
        for filename in os.listdir(path):
            files_list.append(os.path.join(path, filename))
        return files_list


def build_loader_finetune(config, logger):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_modis_dataset(is_train=True, config=config, logger=logger)
    config.freeze()
    dataset_val, _ = build_modis_dataset(is_train=False, config=config, logger=logger)
    logger.info(f"Build dataset: train images = {len(dataset_train)}, val images = {len(dataset_val)}")

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    sampler_train = DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    sampler_val = DistributedSampler(
        dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
    )

    data_loader_train = DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_val = DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
    )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return dataset_train, dataset_val, data_loader_train, data_loader_val, None


def build_dataset(is_train, config, logger):
    transform = build_transform(is_train, config)
    logger.info(f'Fine-tune data transform, is_train={is_train}:\n{transform}')
    
    if config.DATA.DATASET == 'imagenet':
        prefix = 'train' if is_train else 'val'
        root = os.path.join(config.DATA.DATA_PATH, prefix)
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    else:
        raise NotImplementedError("We only support ImageNet Now.")

    return dataset, nb_classes


def build_modis_dataset(is_train, config, logger):
    transform = build_modis_transform()
    logger.info(f'Fine-tune data transform, is_train={is_train}:\n{transform}')
    data_path = config.DATA.DATA_PATH
    split = 'train' if is_train else 'val'
    img_size_singular = config.DATA.IMG_SIZE
    img_size = (img_size_singular, img_size_singular)
    dataset = MODISDataset(data_path, split, img_size, transform=transform)
    nb_classes = 18
    return dataset, nb_classes


def build_modis_transform():
    transform = T.Compose(
                [
                    T.ToTensor(),
                    T.Normalize(
                        mean=[0.0173, 0.0332, 0.0088,
                          0.0136, 0.0381, 0.0348, 0.0249],
                        std=[0.0150, 0.0127, 0.0124,
                         0.0128, 0.0120, 0.0159, 0.0164]
                    ),
                ]
            )
    return transform


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION))
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)