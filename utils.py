import numpy as np
import torch

'''dataset_mean, dataset_std = (0.4914, 0.4822, 0.4465), \
            (0.2470, 0.2435, 0.2616)'''

def calculate_mean_std(dataset):

    #if dataset_mean and dataset_std:
    #    return dataset_std, dataset_std

    imgs = [item[0] for item in dataset]
    imgs = torch.stack(imgs, dim=0)

    mean = []
    std = []
    for i in range(imgs.shape[1]):
        mean.append(imgs[:, i, :, :].mean().item())
        std.append(imgs[:, i, :, :].std().item())

    return tuple(mean), tuple(std)
