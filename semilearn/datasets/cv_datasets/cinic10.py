import torch
import torchvision
import torchvision.transforms as transforms
import os
import json
import numpy as np
import math
from torch.utils.data import ConcatDataset
from torchvision import transforms
from .datasetbase import BasicDataset
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation
from semilearn.datasets.utils import split_ssl_data


cinic_mean = [0.47889522, 0.47227842, 0.43047404]
cinic_std = [0.24205776, 0.23828046, 0.25874835]

def get_cinic10(args, alg, name, num_classes=10, data_dir='./data', include_lb_to_ulb=True, is_all_ulb=False):
    id_classes = args.ID_classes
    id_labels_per_class = args.ID_labels_per_class
    ood_classes = args.OOD_classes
    ood_labels_per_class = args.OOD_labels_per_class
    num_labels = id_labels_per_class * num_classes

    crop_size = args.img_size
    crop_ratio = args.crop_ratio
    
    data_dir = os.path.join(data_dir, name.lower())

    transform_weak = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cinic_mean, cinic_std)
    ])
    transform_medium = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        RandAugment(1, 5),
        transforms.ToTensor(),
        transforms.Normalize(cinic_mean, cinic_std)
    ])
    transform_strong = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 5),
        transforms.ToTensor(),
        transforms.Normalize(cinic_mean, cinic_std)
    ])
    transform_val = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(cinic_mean, cinic_std,)
    ])

    train_set = torchvision.datasets.ImageFolder(data_dir + '/train')  

    eval_set = torchvision.datasets.ImageFolder(data_dir + '/valid')

    data, targets = eval_set.samples, eval_set.targets
    # 划分开集类 验证集
    indices = [i for i, target in enumerate(targets) if target < id_classes]
    data = [data[i] for i in indices]
    targets = [targets[i] for i in indices]

    # print("data shape: ", len(data))

    eval_dset = BasicDataset(alg, data, targets, num_classes, transform_val, False, None, None, False)

    if num_labels != 90000:
        lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(args, data, targets, num_classes, 
                                                                lb_num_labels=num_labels,
                                                                ulb_num_labels=args.ulb_num_labels,
                                                                lb_imbalance_ratio=args.lb_imb_ratio,
                                                                ulb_imbalance_ratio=args.ulb_imb_ratio,
                                                                include_lb_to_ulb=include_lb_to_ulb)
    else:
        lb_data, lb_targets = data, targets
        if include_lb_to_ulb:
            ulb_data, ulb_targets = data, targets

    #切割训练集类别
    indices = [i for i in range(len(lb_targets)) if lb_targets[i] < id_classes]
    lb_data = [lb_data[i] for i in indices]
    lb_targets = [lb_targets[i] for i in indices]

    lb_dset = BasicDataset(alg, lb_data, lb_targets, num_classes, transform_weak, False, transform_strong, transform_strong, False)

    # print("shape", len(lb_targets), len(ulb_targets))
    # print("lb_dset shape", len(lb_dset.targets))

    ulb_dset = BasicDataset(alg, ulb_data, ulb_targets, num_classes, transform_weak, True, transform_medium, transform_strong, False)

    # print("shape", len(lb_targets), len(ulb_targets), len(eval_dset.targets))
    return lb_dset, ulb_dset, eval_dset

