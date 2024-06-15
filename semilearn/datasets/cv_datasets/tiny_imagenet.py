import os
import json
import torchvision
import numpy as np
import math
from PIL import Image

from torchvision import transforms, datasets
from .datasetbase import BasicDataset
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation
from semilearn.datasets.utils import split_ssl_data
from torch.utils.data import Dataset

class ImageFolderInstance(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target, index
    
class TinyImageNetValidation(Dataset):
    def __init__(self, root, label_to_index, transform=None):
        self.root = root
        self.transform = transform
        self.annotations_file = os.path.join(root,  'val_annotations.txt')
        with open(self.annotations_file, 'r') as f:
            self.annotations = f.readlines()
        self.annotations = [line.strip().split('\t') for line in self.annotations]

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
        return image, label, index

mean = [0.4802, 0.4481, 0.3975]
std = [0.2302, 0.2265, 0.2262]

def get_tiny_imagenet(args, alg, name, num_labels, num_classes, data_dir='./data', include_lb_to_ulb=True):
    data_dir = os.path.join(data_dir, name.lower())
    train_path = os.path.join(data_dir, 'train')
    val_path = os.path.join(data_dir, 'val')
    
    train_set = ImageFolderInstance(train_path)
    val_set = ImageFolderInstance(val_path)

    data, targets = train_set.samples, train_set.targets

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
    
    lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(args, data, targets, num_classes,
                                                                lb_num_labels=num_labels,
                                                                ulb_num_labels=args.ulb_num_labels,
                                                                lb_imbalance_ratio=args.lb_imb_ratio,
                                                                ulb_imbalance_ratio=args.ulb_imb_ratio,
                                                                include_lb_to_ulb=include_lb_to_ulb)
