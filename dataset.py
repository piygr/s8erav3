import albumentations as A
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import config as cfg
from utils import calculate_mean_std


class AlbumentationsDataset:
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = np.array(image)
        if self.transform:
            image = self.transform(image=image)["image"]

        label = torch.from_numpy(np.array(label))
        image = torch.from_numpy(image.transpose(2, 0, 1))

        return image, label


def get_transforms(mean, std, p):
    train_transform = A.Compose([
        A.Normalize(mean=mean, std=std),
        A.HorizontalFlip(p=p),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15),
        A.CoarseDropout(max_holes=1, max_height=16, max_width=16,
                        min_holes=1, min_height=16, min_width=16,
                        fill_value=(mean), mask_fill_value=None, p=p),
    ])

    test_transform = A.Compose([
        A.Normalize(mean=mean, std=std)
    ])

    return train_transform, test_transform


def get_dataloaders(batch_size=128, valid_split=0.1, shuffle=True, num_workers=4):
    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transforms.ToTensor())

    mean, std = calculate_mean_std(dataset)

    train_transform, test_transform = get_transforms(mean, std, cfg.transform_probability)

    train_dataset = AlbumentationsDataset(dataset, transform=train_transform)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transforms.ToTensor())
    test_dataset = AlbumentationsDataset(test_dataset, transform=test_transform)

    '''indices = np.arange(len(train_dataset))
    np.random.shuffle(indices)
    split = int(len(train_dataset) * (1 - valid_split))

    train_indices, valid_indices = indices[:split], indices[split:]
    train_subset = Subset(train_dataset, train_indices)
    valid_subset = Subset(train_dataset, valid_indices)
    valid_loader = DataLoader(valid_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    '''

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader
