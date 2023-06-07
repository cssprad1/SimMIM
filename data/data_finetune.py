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

import numpy as np
import random


class MODISDataset(Dataset):

    IMAGE_PATH = os.path.join("images")
    MASK_PATH = os.path.join("labels")

    def __init__(
        self,
        data_paths: list,
        split: str,
        img_size: tuple = (128, 128),
        transform=None,
    ):
        self.img_size = img_size
        self.transform = transform
        self.split = split
        self.data_paths = data_paths
        self.img_list = []
        self.mask_list = []
        for data_path in data_paths:
            img_path = os.path.join(data_path, self.IMAGE_PATH)
            mask_path = os.path.join(data_path, self.MASK_PATH)
            self.img_list.extend(self.get_filenames(img_path))
            self.mask_list.extend(self.get_filenames(mask_path))
        # Split between train and valid set (80/20)

        random_inst = random.Random(12345)  # for repeatability
        n_items = len(self.img_list)
        print(f'Found {n_items} possible patches to use')
        range_n_items = range(n_items)
        range_n_items = random_inst.sample(range_n_items, int(n_items*0.5))
        idxs = set(random_inst.sample(range_n_items, len(range_n_items) // 5))
        total_idxs = set(range_n_items)
        if split == 'train':
            idxs = total_idxs - idxs
        print(f'> Using {len(idxs)} patches for this dataset ({split})')
        self.img_list = [self.img_list[i] for i in idxs]
        self.mask_list = [self.mask_list[i] for i in idxs]
        print(f'>> {split}: {len(self.img_list)}')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx, transpose=True):

        # load image
        img = np.load(self.img_list[idx])
        img = np.clip(img, 0, 1.0)

        # load mask
        mask = np.load(self.mask_list[idx])
        mask = np.expand_dims(mask, axis=0)

        # perform transformations
        img = self.transform(img)

        return img, mask

    def get_filenames(self, path):
        """
        Returns a list of absolute paths to images inside given `path`
        """
        files_list = []
        for filename in sorted(os.listdir(path)):
            files_list.append(os.path.join(path, filename))
        return files_list


def build_loader_finetune(config, logger):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_modis_dataset(
        is_train=True, config=config, logger=logger)
    config.freeze()
    dataset_val, _ = build_modis_dataset(
        is_train=False, config=config, logger=logger)
    logger.info(
        f"Build dataset: train images = {len(dataset_train)}, val images = {len(dataset_val)}")

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    logger.info(f'>>>>>> GLOBAL RANK: {global_rank}')
    logger.info(f'>>>>>> NUM TASKS: {num_tasks}')
    sampler_train = DistributedSampler(
        dataset_train,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True
    )
    sampler_val = DistributedSampler(
        dataset_val,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=False
    )

    data_loader_train = DataLoader(
        dataset_train, config.DATA.BATCH_SIZE, sampler=sampler_train,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    data_loader_val = DataLoader(
        dataset_val, config.DATA.BATCH_SIZE,
        sampler=sampler_val,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return dataset_train, dataset_val, data_loader_train, data_loader_val, None


def build_modis_dataset(is_train, config, logger):
    transform = build_modis_transform()
    logger.info(f'Fine-tune data transform, is_train={is_train}:\n{transform}')
    data_path = config.DATA.DATA_PATH
    split = 'train' if is_train else 'val'
    img_size_singular = config.DATA.IMG_SIZE
    img_size = (img_size_singular, img_size_singular)
    dataset = MODISDataset(data_path, split, img_size,
                           transform=transform)
    nb_classes = 1
    return dataset, nb_classes


def build_modis_transform():
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(
                    mean=[0.15767191, 0.29699719, 0.07437498, 0.12194485, 0.33692781, 0.30693091, 0.22627914],
                    std=[0.14261609, 0.11869394, 0.10452154, 0.1136049,  0.11703701, 0.15358771, 0.1629181]
            ),
        ]
    )
    return transform
