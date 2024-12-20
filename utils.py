import numpy as np

def calculate_mean_std(dataset):
    """Calculate the mean and standard deviation of the dataset."""
    data = np.concatenate([np.array(image) for image, _ in dataset], axis=0)  # Stack all images
    mean = data.mean(axis=(0, 1, 2)) / 255.0  # Normalize to [0.0, 1.0]
    std = data.std(axis=(0, 1, 2)) / 255.0
    return mean, std
