import numpy as np
import torch


def calculate_mean_std(dataset):
    data = np.array(dataset.data / 255.0)
    mean = data.mean(axis=(0, 1, 2))
    std = data.std(axis=(0, 1, 2))
    return mean, std
