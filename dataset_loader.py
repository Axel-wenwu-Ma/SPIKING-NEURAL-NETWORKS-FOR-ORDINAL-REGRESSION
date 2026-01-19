#!/usr/bin/env python3

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')






DATA_ROOT = "data/"


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2023, 0.1994, 0.2010]

class CIFAR10Dataset(Dataset):
    """CIFAR-10 dataset loader"""
    
    def __init__(self, root_dir=None, train=True, transform=None):
        """
        Args:
            root_dir: dataset root directory
            train: whether to load training set
            transform: data transformation
        """
        if root_dir is None:
            root_dir = os.path.join(DATA_ROOT, 'cifar10')
        
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        

        if train:
            self.data, self.targets = self._load_train_data()
        else:
            self.data, self.targets = self._load_test_data()
        

        self.classes = self._load_meta()
        self.num_classes = 10
    
    def _load_train_data(self):

        cifar10_dir = os.path.join(self.root_dir, 'cifar-10-batches-py')
        
        data_list = []
        targets_list = []
        
        for i in range(1, 6):
            file_path = os.path.join(cifar10_dir, f'data_batch_{i}')
            with open(file_path, 'rb') as f:
                batch = pickle.load(f, encoding='bytes')
                data_list.append(batch[b'data'])
                targets_list.extend(batch[b'labels'])
        
        data = np.vstack(data_list).reshape(-1, 3, 32, 32)
        data = data.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        
        return data, np.array(targets_list)
    
    def _load_test_data(self):

        cifar10_dir = os.path.join(self.root_dir, 'cifar-10-batches-py')
        
        file_path = os.path.join(cifar10_dir, 'test_batch')
        with open(file_path, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
            data = batch[b'data'].reshape(-1, 3, 32, 32)
            data = data.transpose(0, 2, 3, 1)
            targets = batch[b'labels']
        
        return data, np.array(targets)
    
    def _load_meta(self):

        cifar10_dir = os.path.join(self.root_dir, 'cifar-10-batches-py')
        
        file_path = os.path.join(cifar10_dir, 'batches.meta')
        with open(file_path, 'rb') as f:
            meta = pickle.load(f, encoding='bytes')
            classes = [name.decode('utf-8') for name in meta[b'label_names']]
        
        return classes
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        target = self.targets[idx]
        

        img = Image.fromarray(img)
        
        if self.transform:
            img = self.transform(img)
        
        return img, target


class CIFAR100Dataset(Dataset):
    """CIFAR-100 dataset loader"""
    
    def __init__(self, root_dir=None, train=True, transform=None, use_coarse_labels=False):
        """
        Args:
            root_dir: dataset root directory
            train: whether to load training set
            transform: data transformation
            use_coarse_labels: whether to use coarse labels (20 classes)
        """
        if root_dir is None:
            root_dir = os.path.join(DATA_ROOT, 'cifar100')
        
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        self.use_coarse_labels = use_coarse_labels
        

        self._load_data()
        

        self._load_meta()
        
        self.num_classes = 20 if use_coarse_labels else 100
    
    def _load_data(self):

        cifar100_dir = os.path.join(self.root_dir, 'cifar-100-python')
        
        if self.train:
            file_path = os.path.join(cifar100_dir, 'train')
        else:
            file_path = os.path.join(cifar100_dir, 'test')
        
        with open(file_path, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
            self.data = batch[b'data'].reshape(-1, 3, 32, 32)
            self.data = self.data.transpose(0, 2, 3, 1)
            
            if self.use_coarse_labels:
                self.targets = np.array(batch[b'coarse_labels'])
            else:
                self.targets = np.array(batch[b'fine_labels'])
    
    def _load_meta(self):

        cifar100_dir = os.path.join(self.root_dir, 'cifar-100-python')
        
        file_path = os.path.join(cifar100_dir, 'meta')
        with open(file_path, 'rb') as f:
            meta = pickle.load(f, encoding='bytes')
            
            if self.use_coarse_labels:
                self.classes = [name.decode('utf-8') for name in meta[b'coarse_label_names']]
            else:
                self.classes = [name.decode('utf-8') for name in meta[b'fine_label_names']]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        target = self.targets[idx]
        

        img = Image.fromarray(img)
        
        if self.transform:
            img = self.transform(img)
        
        return img, target


class ImageNetDataset(Dataset):
    """ImageNet dataset loader"""
    
    def __init__(self, root_dir=None, split='train', transform=None):
        """
        Args:
            root_dir: dataset root directory
            split: 'train' 或 'val'
            transform: data transformation
        """
        if root_dir is None:
            root_dir = os.path.join(DATA_ROOT, 'imagenet1k')
        
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
  
        self.data_dir = os.path.join(root_dir, split)
        

        self.samples = []
        self.targets = []
        self.class_to_idx = {}
        self.classes = []
        
        self._load_dataset()
        self.num_classes = 1000
    
    def _load_dataset(self):
        """Load dataset"""

        class_dirs = sorted([d for d in os.listdir(self.data_dir) 
                           if os.path.isdir(os.path.join(self.data_dir, d))])
        

        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_dirs)}
        self.classes = class_dirs
        

        for class_name, class_idx in self.class_to_idx.items():
            class_dir = os.path.join(self.data_dir, class_name)
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.JPEG')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append(img_path)
                    self.targets.append(class_idx)
        
        self.targets = np.array(self.targets)
        print(f"Loaded {len(self.samples)} images from ImageNet {self.split} set")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        target = self.targets[idx]
        

        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, target


def get_cifar10_dataloaders(batch_size=128, num_workers=4, augment=True, 
                           normalize=True, root_dir=None):
    """

    
    Args:
        batch_size: batch size
        num_workers: number of data loading threads
        augment: whether to use data augmentation
        normalize: whether to normalize
        root_dir: dataset root directory
    
    Returns:
        train_loader, test_loader
    """

    train_transforms_list = []
    if augment:
        train_transforms_list.extend([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
    train_transforms_list.append(transforms.ToTensor())
    if normalize:
        train_transforms_list.append(
            transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD)
        )
    train_transform = transforms.Compose(train_transforms_list)
    

    test_transforms_list = [transforms.ToTensor()]
    if normalize:
        test_transforms_list.append(
            transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD)
        )
    test_transform = transforms.Compose(test_transforms_list)
    

    train_dataset = CIFAR10Dataset(root_dir=root_dir, train=True, transform=train_transform)
    test_dataset = CIFAR10Dataset(root_dir=root_dir, train=False, transform=test_transform)
    

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader



def get_cifar100_dataloaders(batch_size=128, num_workers=4, augment=True, 
                            normalize=True, use_coarse_labels=False, root_dir=None):
    """
    get CIFAR-100 dataloaders
    
    Args:
        batch_size: batch size
        num_workers: number of data loading threads
        augment: whether to use data augmentation
        normalize: whether to normalize
        use_coarse_labels: whether to use coarse labels
        root_dir: dataset root directory

    Returns:
        train_loader, test_loader
    """

    train_transforms_list = []
    if augment:
        train_transforms_list.extend([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
    train_transforms_list.append(transforms.ToTensor())
    if normalize:
        train_transforms_list.append(
            transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD)
        )
    train_transform = transforms.Compose(train_transforms_list)
    

    test_transforms_list = [transforms.ToTensor()]
    if normalize:
        test_transforms_list.append(
            transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD)
        )
    test_transform = transforms.Compose(test_transforms_list)
    

    train_dataset = CIFAR100Dataset(
        root_dir=root_dir, 
        train=True, 
        transform=train_transform,
        use_coarse_labels=use_coarse_labels
    )
    test_dataset = CIFAR100Dataset(
        root_dir=root_dir, 
        train=False, 
        transform=test_transform,
        use_coarse_labels=use_coarse_labels
    )
    

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader



def get_imagenet_dataloaders(batch_size=256, num_workers=8, image_size=224, 
                            augment=True, normalize=True, root_dir=None):
    """
    get ImageNet dataloaders
    
    Args:

        augment: whether to use data augmentation
        normalize: whether to normalize
        root_dir: dataset root directory

    Returns:
        train_loader, val_loader
    """

    train_transforms_list = []
    if augment:
        train_transforms_list.extend([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        ])
    else:
        train_transforms_list.extend([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
        ])
    train_transforms_list.append(transforms.ToTensor())
    if normalize:
        train_transforms_list.append(
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        )
    train_transform = transforms.Compose(train_transforms_list)
    

    val_transforms_list = [
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor()
    ]
    if normalize:
        val_transforms_list.append(
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        )
    val_transform = transforms.Compose(val_transforms_list)
    

    train_dataset = ImageNetDataset(root_dir=root_dir, split='train', transform=train_transform)
    val_dataset = ImageNetDataset(root_dir=root_dir, split='val', transform=val_transform)
    

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, val_loader



class FGNETDataset(Dataset):
    """FGNET face age dataset loader"""
    
    def __init__(self, root_dir=None, train=True, transform=None, train_split=0.8):
        """
        Args:

            train_split: training set ratio
        """
        if root_dir is None:
            root_dir = os.path.join(DATA_ROOT, 'FGNET')
        
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        

        self.samples = []
        self.targets = []
        

        for img_name in sorted(os.listdir(root_dir)):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root_dir, img_name)

                try:
                    age = int(img_name[4:6])
                    self.samples.append(img_path)
                    self.targets.append(age)
                except:
                    continue
        

        self.targets = np.array(self.targets)
        

        n_samples = len(self.samples)
        n_train = int(n_samples * train_split) +1
        

        np.random.seed(42)
        indices = np.random.permutation(n_samples)
        
        if train:
            indices = indices[:n_train]
        else:
            indices = indices[n_train:]
        
        self.samples = [self.samples[i] for i in indices]
        self.targets = self.targets[indices]
        

        self.num_classes = len(np.unique(self.targets))
        print(f"FGNET {'train' if train else 'test'}: {len(self.samples)} images, {self.num_classes} age classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        target = self.targets[idx]
        

        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, target


class HICDataset(Dataset):
    """HIC face age dataset loader"""
    
    def __init__(self, root_dir=None, train=True, transform=None, train_split=0.8):
        """
        Args:
            root_dir: dataset root directory
            train: whether to load training set
            transform: data transformation
            train_split:training set ratio
        """
        if root_dir is None:
            root_dir = os.path.join(DATA_ROOT, 'HIC')
        
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        

        decade_to_label = {
            '1930s': 0,
            '1940s': 1,
            '1950s': 2,
            '1960s': 3,
            '1970s': 4
        }
        
        self.classes = ['1930s', '1940s', '1950s', '1960s', '1970s']
        self.num_classes = 5
        

        self.samples = []
        self.targets = []
        
        for decade, label in decade_to_label.items():
            decade_dir = os.path.join(root_dir, decade)
            if os.path.exists(decade_dir):
                for img_name in os.listdir(decade_dir):
                    if (img_name.lower().endswith(('.jpg', '.jpeg', '.png')) and 
                        not img_name.startswith('.') and 
                        not img_name.startswith('._')):
                        #print (f"Loading image: {img_name} from {decade_dir}")
                        img_path = os.path.join(decade_dir, img_name)
                        self.samples.append(img_path)
                        self.targets.append(label)
        

        self.targets = np.array(self.targets)
        

        n_samples = len(self.samples)
        n_train = int(n_samples * train_split)
        

        np.random.seed(42)
        indices = np.random.permutation(n_samples)
        
        if train:
            indices = indices[:n_train]
        else:
            indices = indices[n_train:]
        
        self.samples = [self.samples[i] for i in indices]
        self.targets = self.targets[indices]
        
        print(f"HIC {'train' if train else 'test'}: {len(self.samples)} images, {self.num_classes} decades")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        target = self.targets[idx]
        

        
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, target


def get_fgnet_dataloaders(batch_size=32, num_workers=4, image_size=224, 
                         augment=True, normalize=True, root_dir=None, train_split=0.8):
    """
    Get FGNET data loaders
    Args:
        batch_size: batch size
        num_workers: number of data loading threads
        image_size: image size
        augment: whether to use data augmentation
        normalize: whether to normalize
        root_dir: dataset root directory
        train_split: training set ratio
    Returns:
        train_loader, test_loader
    """

    train_transforms_list = []
    if augment:
        train_transforms_list.extend([
            #transforms.RandomResizedCrop(image_size),
            #transforms.Resize(round(image_size * 1.1)),
            #transforms.RandomCrop(image_size, padding=4),
            #transforms.RandomHorizontalFlip(),
            #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),

                #transforms.ConvertImageDtype(torch.float32),
                transforms.Resize((int(image_size*1.05), int(image_size*1.05)), antialias=True),
                transforms.RandomCrop((image_size, image_size)),
                transforms.ColorJitter(0.1, 0.1),
                transforms.RandomHorizontalFlip(),
        ])
    else:
        train_transforms_list.extend([
            transforms.Resize((image_size, image_size)),
        ])
    train_transforms_list.append(transforms.ToTensor())
    if normalize:
        train_transforms_list.append(
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        )
    train_transform = transforms.Compose(train_transforms_list)
    

    test_transforms_list = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ]
    if normalize:
        test_transforms_list.append(
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        )
    test_transform = transforms.Compose(test_transforms_list)
    

    train_dataset = FGNETDataset(root_dir=root_dir, train=True, transform=train_transform, train_split=train_split)
    test_dataset = FGNETDataset(root_dir=root_dir, train=False, transform=test_transform, train_split=train_split)
    

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


def get_hic_dataloaders(batch_size=32, num_workers=4, image_size=224, 
                       augment=True, normalize=True, root_dir=None, train_split=0.8):
    """
    Get HIC data loaders
    Args:
        batch_size: batch size
        num_workers: number of data loading threads
        image_size: image size
        augment: whether to use data augmentation
        normalize: whether to normalize
        root_dir: dataset root directory
        train_split: training set ratio
    Returns:
        train_loader, test_loader
    """

    train_transforms_list = []
    if augment:
        train_transforms_list.extend([
            transforms.Resize((int(image_size*1.05), int(image_size*1.05)), antialias=True),
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ])
    else:
        train_transforms_list.extend([
            transforms.Resize((image_size, image_size)),
        ])
    train_transforms_list.append(transforms.ToTensor())
    if normalize:
        train_transforms_list.append(
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        )
    train_transform = transforms.Compose(train_transforms_list)
    

    test_transforms_list = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ]
    if normalize:
        test_transforms_list.append(
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        )
    test_transform = transforms.Compose(test_transforms_list)
    

    train_dataset = HICDataset(root_dir=root_dir, train=True, transform=train_transform, train_split=train_split)
    test_dataset = HICDataset(root_dir=root_dir, train=False, transform=test_transform, train_split=train_split)
    

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader



















def get_dataloader(dataset_name, batch_size=128, num_workers=4, **kwargs):
    """
    Get dataloaders for specified dataset       
    Args:
        dataset_name: dataset name ('cifar10', 'cifar100', 'imagenet', 'fgnet', 'hic')
        batch_size: batch size
        num_workers: number of data loading threads
        **kwargs: additional arguments for specific dataset loaders
    
    Returns:
        train_loader, test_loader, num_classes
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'cifar10':
        train_loader, test_loader = get_cifar10_dataloaders(
            batch_size=batch_size, 
            num_workers=num_workers, 
            **kwargs
        )
        num_classes = 10
        
    elif dataset_name == 'cifar100':
        train_loader, test_loader = get_cifar100_dataloaders(
            batch_size=batch_size, 
            num_workers=num_workers, 
            **kwargs
        )
        num_classes = 100 if not kwargs.get('use_coarse_labels', False) else 20
        
    elif dataset_name == 'imagenet':
        train_loader, test_loader = get_imagenet_dataloaders(
            batch_size=batch_size, 
            num_workers=num_workers, 
            **kwargs
        )
        num_classes = 1000
    elif dataset_name == 'fgnet':
        train_loader, test_loader = get_fgnet_dataloaders(
            batch_size=batch_size, 
            num_workers=num_workers, 
            **kwargs
        )
        # 获取实际的类别数
        num_classes = 70
    elif dataset_name == 'hic':
        train_loader, test_loader = get_hic_dataloaders(
            batch_size=batch_size, 
            num_workers=num_workers, 
            **kwargs
        )
        num_classes = 5
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return train_loader, test_loader, num_classes




