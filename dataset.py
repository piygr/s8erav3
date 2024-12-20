import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import numpy as np
from torchvision import datasets
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
        image = np.array(image)  # Convert PIL Image to NumPy array
        if self.transform:
            augmented = self.transform(image=image)  # Albumentations transform
            image = augmented["image"]  # Extract the transformed image
        label = torch.tensor(label, dtype=torch.long)  # Convert label to tensor
        return image, label


def get_transforms(mean, std, p):
    # Ensure mean and std are lists of floats
    mean = mean.tolist()
    std = std.tolist()

    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=p),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=p),
            A.CoarseDropout(
                max_holes=1,
                max_height=16,
                max_width=16,
                min_holes=1,
                min_height=16,
                min_width=16,
                fill_value=mean,  # Compatible with Albumentations (list of floats)
                mask_fill_value=None,
                p=p,
            ),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),  # Normalize with Albumentations
            ToTensorV2(),  # Converts image to PyTorch tensor and reorders to (C, H, W)
        ]
    )

    test_transform = A.Compose(
        [
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )

    return train_transform, test_transform


def get_dataloaders(batch_size=128, valid_split=0.1, shuffle=True, num_workers=4):
    # Load raw CIFAR-10 dataset (no ToTensor() here, as Albumentations handles it)
    dataset = datasets.CIFAR10(root="./data", train=True, download=True)

    # Calculate mean and std from the raw dataset
    mean, std = calculate_mean_std(dataset)

    # Get Albumentations transforms
    train_transform, test_transform = get_transforms(mean, std, cfg.transform_probability)

    # Create Albumentations-wrapped datasets
    train_dataset = AlbumentationsDataset(dataset, transform=train_transform)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True)
    test_dataset = AlbumentationsDataset(test_dataset, transform=test_transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

