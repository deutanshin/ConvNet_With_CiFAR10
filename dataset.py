import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np

def get_dataloaders(batch_size=128, overfit=False, num_overfit_samples=1000, num_overfit_valid=500):
    # Preprocess unit
    base_preprocess = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]


    # Loading dataset
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose(base_preprocess))

    if overfit:
        preprocess = transforms.Compose(base_preprocess)

        full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=preprocess)
    else:
        preprocess = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
        ] + base_preprocess)

        full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=preprocess)


    # split dataset train/val
    if overfit:
        # overfit mode, Use 1000:500
        train_indices = range(num_overfit_samples)
        val_indices = range(num_overfit_samples, num_overfit_samples + num_overfit_valid)
        
        train_dataset = Subset(full_train_dataset, train_indices)
        val_dataset = Subset(full_train_dataset, val_indices)
    else:
        # none overfit mode, Use 9:1 randomly
        train_size = int(0.9 * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # 4. Data Loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader