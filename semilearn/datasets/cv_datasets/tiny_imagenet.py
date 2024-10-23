import os
import json
import torchvision
import numpy as np
import math
from PIL import Image
import torch
from torchvision import transforms, datasets
from .datasetbase import BasicDataset
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation
from semilearn.datasets.utils import split_ssl_data
from torch.utils.data import Dataset

class ImageFolderInstance(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        # 返回三通道图片的(32,32,3)数组
        img = transforms.ToTensor()(img)
        img = self.loader(path)
        return img, target, index
    
class TinyImageNetValidation(Dataset):
    def __init__(self, root, label_to_index, seen_classes=None, transform=None):
        self.root = root
        self.transform = transform
        self.annotations_file = os.path.join(root,  'val_annotations.txt')
        with open(self.annotations_file, 'r') as f:
            self.annotations = f.readlines()
        self.annotations = [line.strip().split('\t') for line in self.annotations]
        indices = [i for i, ann in enumerate(self.annotations) if label_to_index[ann[1]] in seen_classes]
        self.annotations = [self.annotations[i] for i in indices]

        # prepare a mapping from class name to index
        self.label_to_index = label_to_index

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        image_path = os.path.join(self.root, 'images', self.annotations[index][0])
        image = Image.open(image_path).convert('RGB')
        label = int(self.label_to_index[self.annotations[index][1]])
        if self.transform is not None:
            image = self.transform(image)
        # return image, label, index
        return {'idx_lb': index, 'x_lb': image, 'y_lb': label}

mean, std = {}, {}
mean['tiny_imagenet'] = [0.4802, 0.4481, 0.3975]
std['tiny_imagenet'] = [0.2770, 0.2691, 0.2821]

def get_tiny_imagenet(args, alg, name, num_classes=200, data_dir='./data', include_lb_to_ulb=True,use_val=True):
    id_classes = args.ID_classes
    id_labels_per_class = args.ID_labels_per_class
    ood_classes = args.OOD_classes
    ood_labels_per_class = args.OOD_labels_per_class

    seed = args.seed

    if use_val:
        num_labels = id_labels_per_class * num_classes
        seen_classes = np.random.RandomState(seed).choice(num_classes, id_classes, replace=False)
        seen_labels_per_class = id_labels_per_class
    else:
        num_labels = ood_labels_per_class * num_classes
        seen_classes = np.random.RandomState(seed).choice(num_classes, ood_classes, replace=False)
        seen_labels_per_class = ood_labels_per_class
        if seen_classes == 0:
            return None, None, None

    data_dir = os.path.join(data_dir, name.lower())
    train_path = os.path.join(data_dir, 'train')
    val_path = os.path.join(data_dir, 'val')
    
    train_set = ImageFolderInstance(train_path)

    label_to_index = train_set.class_to_idx
    

    data, targets = train_set.samples, train_set.targets
    
    
    # print("data: ", data[500])
    # print("targets: ", targets[500])
    # print("data shape: ", len(data))

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
    
    val_set = TinyImageNetValidation(val_path, label_to_index, seen_classes, transform=transform_val)
    # print(f"seen_classes: {seen_classes}")
    # print(f"val_set: {len(val_set)}")

    
    if seen_labels_per_class != 500:
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
        else:
            ulb_data, ulb_targets = None, None

    # 划分开集类
    indices = [i for i, target in enumerate(targets) if target in seen_classes]
    data = [data[i] for i in indices]
    targets = [targets[i] for i in indices]

    if use_val:
        lb_dset = BasicDataset(alg, lb_data, lb_targets, num_classes, transform_weak, False, transform_strong, transform_strong, False, data_type='pil')
        ulb_dset = BasicDataset(alg, ulb_data, ulb_targets, num_classes, transform_weak, True, transform_medium, transform_strong, False, data_type='pil')
    else:
        lb_dset = None
        ulb_dset = BasicDataset(alg, lb_data, lb_targets, num_classes, transform_weak, True, transform_medium, transform_strong, False, data_type='pil')
 
    return lb_dset, ulb_dset, val_set