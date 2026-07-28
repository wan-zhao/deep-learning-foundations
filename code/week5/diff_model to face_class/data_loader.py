import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os

def get_data_loaders(data_dir, batch_size, split_ratio=0.8):
    """
    Creates train and validation data loaders.
    
    Args:
        data_dir (str): Path to the dataset directory (containing class folders).
        batch_size (int): Batch size for the loaders.
        split_ratio (float): Ratio of training data (default 0.8).
        
    Returns:
        train_loader, val_loader, class_names
    """
    
    # Define transforms
    # Resize to 100x100, Convert to Tensor, Normalize
    # Normalization values (0.5, 0.5, 0.5) are standard for 3-channel images if no specific stats are known
    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Load dataset
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    class_names = full_dataset.classes
    
    # Split dataset
    train_size = int(split_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, class_names
