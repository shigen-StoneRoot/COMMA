from monai import data
import random
import os
from monai.data.utils import pad_list_data_collate
from monai.data import ImageDataset
import matplotlib.pyplot as plt

import math
import warnings
from typing import List
from torch.optim import Optimizer
import torch
from typing import Any, Callable, List, Sequence, Tuple, Union
from monai.utils import BlendMode, PytorchPadMode, fall_back_tuple, look_up_option
import torch.nn.functional as F

from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    RandRotate90,
    RandSpatialCrop,
    ScaleIntensity,
)


# def load_dataset(ds_transform, img_list, gt_list, shuffle=False, seed=123):
#
#     train_ds_list = [{'image': img_path,
#                       'label': gt_path} for img_path, gt_path in zip(img_list, gt_list)]
#
#     if shuffle:
#         random.seed(seed)
#         random.shuffle(train_ds_list)
#
#     train_dataset = data.Dataset(data=train_ds_list, transform=ds_transform)
#
#     return train_dataset

def load_dataloader(ds_transform, img_list, gt_list, shuffle=False, seed=123, batch_size=2,
                    pin_memory=False, persistent_workers=True):

    train_dataset = load_dataset(ds_transform, img_list, gt_list, shuffle=False, seed=seed)
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=pad_list_data_collate
    )

    return train_loader

def load_dataset(ds_transform, img_list, gt_list, shuffle=False, seed=123):

    train_ds_list = [{'image': img_path,
                      'label': gt_path} for img_path, gt_path in zip(img_list, gt_list)]

    if shuffle:
        random.seed(seed)
        random.shuffle(train_ds_list)

    train_dataset = data.Dataset(data=train_ds_list, transform=ds_transform)

    return train_dataset

def load_GL_dataloader(ds_transform, img_list, gt_list, shuffle=False, seed=123, batch_size=2,
                       pin_memory=False, persistent_workers=True, num_workers=4):

    ds_list = [{'image': img_path,
                'label': gt_path,
                'whole': img_path} for img_path, gt_path in zip(img_list, gt_list)]
    dataset = data.Dataset(data=ds_list, transform=ds_transform)
    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=pad_list_data_collate
    )

    return loader

def save_checkpoint(model, epoch, args, filename="model.pt", optimizer=None, scheduler=None):
    state_dict = model.state_dict()
    save_dict = {"epoch": epoch, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.ckdir, str(epoch) + '_' + filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


if __name__ == '__main__':
    from monai import transforms


    train_transform = transforms.Compose([
        transforms.LoadImaged(keys=["image", "label", "whole"], image_only=True),
        transforms.EnsureChannelFirstd(keys=["image", "label", "whole"], channel_dim='no_channel'),
        transforms.NormalizeIntensityd(keys=["image", "whole"], nonzero=True),
        transforms.RandSpatialCropd(keys=["image", "label"], roi_size=(96, 96, 96), random_size=False),
        transforms.Resized(keys=["whole"], mode='trilinear', spatial_size=(160, 160, 64)),
        transforms.ToTensord(keys=["image", "label", "whole"])
    ])

    img_dir = r'/home/genshi/nnUNet/nnUNet_raw_data_base/Dataset301_TubeTK/imagesTr'
    gt_dir = r'/home/genshi/nnUNet/nnUNet_raw_data_base/Dataset301_TubeTK/labelsTr'
    subs = os.listdir(gt_dir)

    img_list = [os.path.join(img_dir, sub.replace('.nii.gz', '_0000.nii.gz')) for sub in subs]
    gt_list = [os.path.join(gt_dir, sub) for sub in subs]

    # train_loader = load_dataloader(train_transform, img_list, gt_list,
    #                                shuffle=True, seed=123, batch_size=1,
    #                                pin_memory=True, persistent_workers=True)
    train_loader = load_GL_dataloader(train_transform, img_list, gt_list,
                                      shuffle=True, seed=123, batch_size=1,
                                      pin_memory=True, persistent_workers=True, num_workers=4)
    for batch_data in train_loader:
        img, gt, whole = batch_data['image'], batch_data['label'], batch_data['whole']
        plt.imshow(img.numpy().squeeze().max(2), cmap='gray')
        plt.show()
        plt.imshow(gt.numpy().squeeze().max(2), cmap='gray')
        plt.show()
        plt.imshow(whole.numpy().squeeze().max(2), cmap='gray')
        plt.show()
        break



