# s8erav3: Deep Learning on CIFAR-10 with Albumentations and PyTorch

## Overview

`s8erav3` is a deep learning project designed to classify images from the CIFAR-10 dataset. The project leverages PyTorch for model implementation and training, and the `albumentations` library for advanced data augmentation. It features a custom CNN model with configurable convolutional blocks, including support for depthwise separable convolutions and dilated convolutions.

## Project Structure

```
s8erav3/
├── config.py                # Configurations for training
├── dataset.py               # Dataset loading and augmentation
├── model.py                 # Custom CNN architecture
├── train.py                 # Training script
├── utils.py                 # Utility functions
├── requirements.txt         # Required libraries
└── README.md                # Project documentation
```

## Project Features

- **Custom Convolutional Neural Network**:
  - Flexible convolutional blocks with support for depthwise separable and dilated convolutions.
  - Includes dropout layers for regularization and batch normalization for stable training.
  - Designed to increase the receptive field while minimizing parameter count.

- **Data Augmentation with Albumentations**:
  - Advanced augmentations like horizontal flip, shift-scale-rotate, and coarse dropout.
  - Normalization applied based on dataset statistics.

- **CIFAR-10 Dataset**:
  - 10-class image classification task with 32x32 RGB images.
  - Training and test datasets are preprocessed and augmented to improve generalization.

- **Training Pipeline**:
  - Supports dynamic training configurations (batch size, number of workers, etc.).
  - Provides validation support for performance monitoring.

## Model Architecture

The model is composed of convolutional blocks (`Block`) and a final classifier (`Net`). Key features include:
- **Block**:
  - Depthwise separable and dilated convolutions for efficiency and receptive field growth.
  - Configurable number of layers (1-3) per block.
- **Net**:
  - Four convolutional blocks followed by global average pooling and a linear classifier.
  - Output: 10-class log-probabilities using `F.log_softmax`.

### Receptive Field (RF) Growth
The model is carefully designed to ensure consistent growth of the receptive field:
1. Block 1: RF grows to 9 (includes dilated convolution).
2. Block 2: RF grows to 15.
3. Block 3: RF grows to 27.
4. Block 4: RF grows to 51 (final receptive field).

### Model Summary

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 32, 32]             448
       BatchNorm2d-2           [-1, 16, 32, 32]              32
         Dropout2d-3           [-1, 16, 32, 32]               0
            Conv2d-4           [-1, 16, 32, 32]           2,320
       BatchNorm2d-5           [-1, 16, 32, 32]              32
         Dropout2d-6           [-1, 16, 32, 32]               0
            Conv2d-7           [-1, 16, 32, 32]           2,320
       BatchNorm2d-8           [-1, 16, 32, 32]              32
         Dropout2d-9           [-1, 16, 32, 32]               0
           Conv2d-10           [-1, 16, 32, 32]             160
           Conv2d-11           [-1, 32, 32, 32]             544
      BatchNorm2d-12           [-1, 32, 32, 32]              64
        Dropout2d-13           [-1, 32, 32, 32]               0
           Conv2d-14           [-1, 32, 32, 32]           9,248
      BatchNorm2d-15           [-1, 32, 32, 32]              64
        Dropout2d-16           [-1, 32, 32, 32]               0
           Conv2d-17           [-1, 32, 16, 16]           9,248
      BatchNorm2d-18           [-1, 32, 16, 16]              64
        Dropout2d-19           [-1, 32, 16, 16]               0
           Conv2d-20           [-1, 32, 16, 16]             320
           Conv2d-21           [-1, 64, 16, 16]           2,112
      BatchNorm2d-22           [-1, 64, 16, 16]             128
        Dropout2d-23           [-1, 64, 16, 16]               0
           Conv2d-24           [-1, 64, 16, 16]          36,928
      BatchNorm2d-25           [-1, 64, 16, 16]             128
        Dropout2d-26           [-1, 64, 16, 16]               0
           Conv2d-27             [-1, 64, 8, 8]          36,928
      BatchNorm2d-28             [-1, 64, 8, 8]             128
        Dropout2d-29             [-1, 64, 8, 8]               0
           Conv2d-30             [-1, 32, 8, 8]          18,464
      BatchNorm2d-31             [-1, 32, 8, 8]              64
        Dropout2d-32             [-1, 32, 8, 8]               0
           Conv2d-33             [-1, 32, 8, 8]           9,248
      BatchNorm2d-34             [-1, 32, 8, 8]              64
        Dropout2d-35             [-1, 32, 8, 8]               0
           Conv2d-36             [-1, 32, 6, 6]           9,248
        AvgPool2d-37             [-1, 32, 1, 1]               0
           Conv2d-38             [-1, 10, 1, 1]             330
================================================================
Total params: 138,666
Trainable params: 138,666
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 3.95
Params size (MB): 0.53
Estimated Total Size (MB): 4.49
```

## Installation

### Prerequisites

1. Python 3.7 or higher.
2. A CUDA-capable GPU (optional, but recommended).
3. Required libraries (install via `requirements.txt`):
   ```bash
   pip install -r requirements.txt
   ```

### Repository Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/s8erav3.git
   cd s8erav3
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

Run the training script to train the model:
```bash
python train.py
```

The script will:
- Load the CIFAR-10 dataset.
- Apply data augmentations using Albumentations.
- Train the `Net` model and display training/validation metrics.

### Configuration

Modify `config.py` to adjust training parameters like batch size, learning rate, and augmentation probabilities.

### Model File

The CNN architecture is implemented in `model.py` and supports customization of block configurations.


## Results

Training on CIFAR-10 achieves high accuracy with the use of advanced data augmentation and a carefully designed model architecture.

| Metric          | Value           |
|------------------|-----------------|
| Training Accuracy| ~82% (after 30 epochs) |
| Test Accuracy    | ~85%           |

### Logs

```
Epoch 1/100 - Training: 100%|█| 391/391 [01:06<00:00,  5.84it/s, accuracy=33.766
Epoch 1/100 - Validation: 100%|█| 79/79 [00:01<00:00, 59.87it/s, accuracy=47.160

Epoch 1/100 Summary:
Train Loss: 1.7405, Train Acc: 33.7660
Valid Loss: 1.4138, Valid Acc: 47.1600
Epoch 2/100 - Training: 100%|█| 391/391 [01:06<00:00,  5.84it/s, accuracy=48.748
Epoch 2/100 - Validation: 100%|█| 79/79 [00:01<00:00, 61.56it/s, accuracy=58.030

Epoch 2/100 Summary:
Train Loss: 1.3887, Train Acc: 48.7480
Valid Loss: 1.1745, Valid Acc: 58.0300
Epoch 3/100 - Training: 100%|█| 391/391 [01:06<00:00,  5.86it/s, accuracy=55.942
Epoch 3/100 - Validation: 100%|█| 79/79 [00:01<00:00, 61.56it/s, accuracy=64.580

Epoch 3/100 Summary:
Train Loss: 1.2069, Train Acc: 55.9420
Valid Loss: 0.9922, Valid Acc: 64.5800
Epoch 4/100 - Training: 100%|█| 391/391 [01:04<00:00,  6.02it/s, accuracy=60.792
Epoch 4/100 - Validation: 100%|█| 79/79 [00:01<00:00, 55.49it/s, accuracy=66.850

Epoch 4/100 Summary:
Train Loss: 1.0823, Train Acc: 60.7920
Valid Loss: 0.9124, Valid Acc: 66.8500
Epoch 5/100 - Training: 100%|█| 391/391 [01:08<00:00,  5.68it/s, accuracy=64.386
Epoch 5/100 - Validation: 100%|█| 79/79 [00:01<00:00, 62.59it/s, accuracy=70.310

Epoch 5/100 Summary:
Train Loss: 0.9868, Train Acc: 64.3860
Valid Loss: 0.8301, Valid Acc: 70.3100
Epoch 6/100 - Training: 100%|█| 391/391 [01:07<00:00,  5.80it/s, accuracy=67.086
Epoch 6/100 - Validation: 100%|█| 79/79 [00:01<00:00, 62.40it/s, accuracy=73.760

Epoch 6/100 Summary:
Train Loss: 0.9270, Train Acc: 67.0860
Valid Loss: 0.7559, Valid Acc: 73.7600
Epoch 7/100 - Training: 100%|█| 391/391 [01:05<00:00,  6.00it/s, accuracy=68.990
Epoch 7/100 - Validation: 100%|█| 79/79 [00:01<00:00, 59.56it/s, accuracy=74.980

Epoch 7/100 Summary:
Train Loss: 0.8763, Train Acc: 68.9900
Valid Loss: 0.7117, Valid Acc: 74.9800
Epoch 8/100 - Training: 100%|█| 391/391 [01:07<00:00,  5.82it/s, accuracy=70.670
Epoch 8/100 - Validation: 100%|█| 79/79 [00:01<00:00, 61.20it/s, accuracy=75.950

Epoch 8/100 Summary:
Train Loss: 0.8338, Train Acc: 70.6700
Valid Loss: 0.6877, Valid Acc: 75.9500
Epoch 9/100 - Training: 100%|█| 391/391 [01:06<00:00,  5.85it/s, accuracy=71.614
Epoch 9/100 - Validation: 100%|█| 79/79 [00:01<00:00, 61.84it/s, accuracy=76.930

Epoch 9/100 Summary:
Train Loss: 0.8055, Train Acc: 71.6140
Valid Loss: 0.6648, Valid Acc: 76.9300
Epoch 10/100 - Training: 100%|█| 391/391 [01:07<00:00,  5.83it/s, accuracy=72.75
Epoch 10/100 - Validation: 100%|█| 79/79 [00:01<00:00, 60.82it/s, accuracy=78.00

Epoch 10/100 Summary:
Train Loss: 0.7769, Train Acc: 72.7580
Valid Loss: 0.6345, Valid Acc: 78.0000
Epoch 11/100 - Training: 100%|█| 391/391 [01:07<00:00,  5.79it/s, accuracy=73.79
Epoch 11/100 - Validation: 100%|█| 79/79 [00:01<00:00, 60.45it/s, accuracy=78.59

Epoch 11/100 Summary:
Train Loss: 0.7495, Train Acc: 73.7900
Valid Loss: 0.6170, Valid Acc: 78.5900
Epoch 12/100 - Training: 100%|█| 391/391 [01:07<00:00,  5.83it/s, accuracy=74.62
Epoch 12/100 - Validation: 100%|█| 79/79 [00:01<00:00, 61.87it/s, accuracy=80.19

Epoch 12/100 Summary:
Train Loss: 0.7233, Train Acc: 74.6260
Valid Loss: 0.5735, Valid Acc: 80.1900
Epoch 13/100 - Training: 100%|█| 391/391 [01:06<00:00,  5.88it/s, accuracy=75.18
Epoch 13/100 - Validation: 100%|█| 79/79 [00:01<00:00, 61.97it/s, accuracy=79.61

Epoch 13/100 Summary:
Train Loss: 0.7131, Train Acc: 75.1820
Valid Loss: 0.5872, Valid Acc: 79.6100
Epoch 14/100 - Training: 100%|█| 391/391 [01:05<00:00,  5.98it/s, accuracy=75.52
Epoch 14/100 - Validation: 100%|█| 79/79 [00:01<00:00, 60.73it/s, accuracy=80.42

Epoch 14/100 Summary:
Train Loss: 0.6964, Train Acc: 75.5240
Valid Loss: 0.5762, Valid Acc: 80.4200
Epoch 15/100 - Training: 100%|█| 391/391 [01:09<00:00,  5.64it/s, accuracy=76.19
Epoch 15/100 - Validation: 100%|█| 79/79 [00:01<00:00, 60.06it/s, accuracy=80.51

Epoch 15/100 Summary:
Train Loss: 0.6775, Train Acc: 76.1920
Valid Loss: 0.5693, Valid Acc: 80.5100
Epoch 16/100 - Training: 100%|█| 391/391 [01:05<00:00,  5.96it/s, accuracy=76.61
Epoch 16/100 - Validation: 100%|█| 79/79 [00:01<00:00, 59.85it/s, accuracy=81.58

Epoch 16/100 Summary:
Train Loss: 0.6678, Train Acc: 76.6140
Valid Loss: 0.5341, Valid Acc: 81.5800
Epoch 17/100 - Training: 100%|█| 391/391 [01:06<00:00,  5.88it/s, accuracy=77.10
Epoch 17/100 - Validation: 100%|█| 79/79 [00:01<00:00, 60.22it/s, accuracy=81.80

Epoch 17/100 Summary:
Train Loss: 0.6581, Train Acc: 77.1000
Valid Loss: 0.5334, Valid Acc: 81.8000
Epoch 18/100 - Training: 100%|█| 391/391 [01:08<00:00,  5.73it/s, accuracy=77.46
Epoch 18/100 - Validation: 100%|█| 79/79 [00:01<00:00, 59.92it/s, accuracy=82.47

Epoch 18/100 Summary:
Train Loss: 0.6433, Train Acc: 77.4620
Valid Loss: 0.5041, Valid Acc: 82.4700
Epoch 19/100 - Training: 100%|█| 391/391 [01:04<00:00,  6.07it/s, accuracy=78.18
Epoch 19/100 - Validation: 100%|█| 79/79 [00:01<00:00, 58.76it/s, accuracy=82.41

Epoch 19/100 Summary:
Train Loss: 0.6265, Train Acc: 78.1800
Valid Loss: 0.5076, Valid Acc: 82.4100
Epoch 20/100 - Training: 100%|█| 391/391 [01:07<00:00,  5.82it/s, accuracy=78.22
Epoch 20/100 - Validation: 100%|█| 79/79 [00:01<00:00, 60.67it/s, accuracy=82.73

Epoch 20/100 Summary:
Train Loss: 0.6199, Train Acc: 78.2240
Valid Loss: 0.5087, Valid Acc: 82.7300
Epoch 21/100 - Training: 100%|█| 391/391 [01:06<00:00,  5.87it/s, accuracy=78.75
Epoch 21/100 - Validation: 100%|█| 79/79 [00:01<00:00, 61.39it/s, accuracy=82.53

Epoch 21/100 Summary:
Train Loss: 0.6102, Train Acc: 78.7560
Valid Loss: 0.5190, Valid Acc: 82.5300
Epoch 22/100 - Training: 100%|█| 391/391 [01:07<00:00,  5.78it/s, accuracy=78.98
Epoch 22/100 - Validation: 100%|█| 79/79 [00:01<00:00, 61.22it/s, accuracy=83.28

Epoch 22/100 Summary:
Train Loss: 0.6028, Train Acc: 78.9860
Valid Loss: 0.4890, Valid Acc: 83.2800
Epoch 23/100 - Training: 100%|█| 391/391 [01:08<00:00,  5.68it/s, accuracy=79.35
Epoch 23/100 - Validation: 100%|█| 79/79 [00:01<00:00, 61.32it/s, accuracy=83.09

Epoch 23/100 Summary:
Train Loss: 0.5945, Train Acc: 79.3540
Valid Loss: 0.4926, Valid Acc: 83.0900
Epoch 24/100 - Training: 100%|█| 391/391 [01:06<00:00,  5.91it/s, accuracy=79.37
Epoch 24/100 - Validation: 100%|█| 79/79 [00:01<00:00, 59.73it/s, accuracy=82.68

Epoch 24/100 Summary:
Train Loss: 0.5885, Train Acc: 79.3700
Valid Loss: 0.5139, Valid Acc: 82.6800
Epoch 25/100 - Training: 100%|█| 391/391 [01:09<00:00,  5.63it/s, accuracy=81.27
Epoch 25/100 - Validation: 100%|█| 79/79 [00:01<00:00, 57.88it/s, accuracy=84.79

Epoch 25/100 Summary:
Train Loss: 0.5353, Train Acc: 81.2700
Valid Loss: 0.4380, Valid Acc: 84.7900
Epoch 26/100 - Training: 100%|█| 391/391 [01:05<00:00,  5.97it/s, accuracy=81.95
Epoch 26/100 - Validation: 100%|█| 79/79 [00:01<00:00, 57.49it/s, accuracy=84.93

Epoch 26/100 Summary:
Train Loss: 0.5205, Train Acc: 81.9560
Valid Loss: 0.4342, Valid Acc: 84.9300
Epoch 27/100 - Training: 100%|█| 391/391 [01:06<00:00,  5.87it/s, accuracy=81.90
Epoch 27/100 - Validation: 100%|█| 79/79 [00:01<00:00, 62.13it/s, accuracy=85.16

Epoch 27/100 Summary:
Train Loss: 0.5156, Train Acc: 81.9060
Valid Loss: 0.4303, Valid Acc: 85.1600
Epoch 28/100 - Training: 100%|█| 391/391 [01:03<00:00,  6.20it/s, accuracy=82.38
Epoch 28/100 - Validation: 100%|█| 79/79 [00:01<00:00, 61.48it/s, accuracy=85.25

Epoch 28/100 Summary:
Train Loss: 0.5059, Train Acc: 82.3880
Valid Loss: 0.4284, Valid Acc: 85.2500
Epoch 29/100 - Training: 100%|█| 391/391 [01:08<00:00,  5.72it/s, accuracy=82.34
Epoch 29/100 - Validation: 100%|█| 79/79 [00:01<00:00, 60.76it/s, accuracy=85.10

Epoch 29/100 Summary:
Train Loss: 0.5052, Train Acc: 82.3460
Valid Loss: 0.4288, Valid Acc: 85.1000
Epoch 30/100 - Training: 100%|█| 391/391 [01:05<00:00,  5.94it/s, accuracy=82.36
Epoch 30/100 - Validation: 100%|█| 79/79 [00:01<00:00, 60.39it/s, accuracy=85.52

Epoch 30/100 Summary:
Train Loss: 0.5050, Train Acc: 82.3640
Valid Loss: 0.4218, Valid Acc: 85.5200
Epoch 31/100 - Training: 100%|█| 391/391 [01:06<00:00,  5.87it/s, accuracy=82.31
Epoch 31/100 - Validation: 100%|█| 79/79 [00:01<00:00, 59.92it/s, accuracy=85.44

Epoch 31/100 Summary:
Train Loss: 0.5026, Train Acc: 82.3120
Valid Loss: 0.4257, Valid Acc: 85.4400
```
*Note: Results may vary based on hardware and hyperparameters.*

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Acknowledgments

- [PyTorch](https://pytorch.org) for deep learning framework.
- [Albumentations](https://albumentations.ai/) for data augmentation.
- CIFAR-10 dataset by [Alex Krizhevsky](https://www.cs.toronto.edu/~kriz/cifar.html).

