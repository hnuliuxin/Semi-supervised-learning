# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import torchvision
import numpy as np
import math
from torch.utils.data import ConcatDataset
from torchvision import transforms
from .datasetbase import BasicDataset
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation
from semilearn.datasets.utils import split_ssl_data


mean, std = {}, {}
mean['cifar10'] = [0.485, 0.456, 0.406]
mean['cifar100'] = [x / 255 for x in [129.3, 124.1, 112.4]]

std['cifar10'] = [0.229, 0.224, 0.225]
std['cifar100'] = [x / 255 for x in [68.2, 65.4, 70.4]]

  




def get_cifar(args, alg, name, num_classes=100, data_dir='./data', include_lb_to_ulb=True, use_val=True):
    id_classes = args.ID_classes
    id_labels_per_class = args.ID_labels_per_class
    ood_classes = args.OOD_classes
    ood_labels_per_class = args.OOD_labels_per_class
    if use_val:
        num_labels = id_labels_per_class * num_classes
        seen_classes = id_classes
        seen_labels_per_class = id_labels_per_class
    else:
        num_labels = ood_labels_per_class * num_classes
        seen_classes = ood_classes
        seen_labels_per_class = ood_labels_per_class

    data_dir = os.path.join(data_dir, name.lower())
    dset = getattr(torchvision.datasets, name.upper())
    dset = dset(data_dir, train=True, download=True)
    data, targets = dset.data, dset.targets
    crop_size = args.img_size
    crop_ratio = args.crop_ratio

    transform_weak = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])
    transform_medium = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        RandAugment(1, 5),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])
    transform_strong = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 5),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])
    transform_val = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name],)
    ])
  


    dset = getattr(torchvision.datasets, name.upper())
    dset = dset(data_dir, train=False, download=True)
    test_data, test_targets = dset.data, dset.targets
    # 切割验证集类别
    
    indices = [i for i in range(len(test_targets)) if test_targets[i] < seen_classes] 
    test_data = [test_data[i] for i in indices]
    test_targets = [test_targets[i] for i in indices]

    eval_dset = BasicDataset(alg, test_data, test_targets, num_classes, transform_val, False, None, None, False)

    # print("eval_dset前五个标签：", eval_dset.targets[:5])
    
    if seen_labels_per_class != 500:
        lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(args, data, targets, num_classes, 
                                                                lb_num_labels=num_labels,
                                                                ulb_num_labels=args.ulb_num_labels,
                                                                lb_imbalance_ratio=args.lb_imb_ratio,
                                                                ulb_imbalance_ratio=args.ulb_imb_ratio,
                                                                include_lb_to_ulb=include_lb_to_ulb)
    else:
        lb_data, lb_targets= data, targets
        if include_lb_to_ulb:
            ulb_data, ulb_targets = data, targets
        else:
            ulb_data, ulb_targets = None, None

    #切割训练集类别
    indices = [i for i in range(len(lb_targets)) if lb_targets[i] < seen_classes]   
    lb_data = [lb_data[i] for i in indices]
    lb_targets = [lb_targets[i] for i in indices]
    if use_val:
        lb_dset = BasicDataset(alg, lb_data, lb_targets, num_classes, transform_weak, False, transform_strong, transform_strong, False)
        ulb_dset = BasicDataset(alg, ulb_data, ulb_targets, num_classes, transform_weak, True, transform_medium, transform_strong, False)
    else:
        lb_dset = None
        ulb_dset = BasicDataset(alg, lb_data, lb_targets, num_classes, transform_weak, True, transform_medium, transform_strong, False)
    # print("shape", len(lb_targets), len(ulb_targets), len(eval_dset.targets))
    return lb_dset, ulb_dset, eval_dset
