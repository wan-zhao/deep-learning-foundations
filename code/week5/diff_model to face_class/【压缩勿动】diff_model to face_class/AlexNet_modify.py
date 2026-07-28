import torch
from torch import nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=2, dropout=0.5):  # ← 添加 dropout 参数
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # Conv 1
            nn.Conv2d(3, 96, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(96), # Added BN
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv 2
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256), # Added BN
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv 3
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384), # Added BN
            nn.ReLU(),
            
            # Conv 4
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384), # Added BN
            nn.ReLU(),
            
            # Conv 5
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), # Added BN
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Reduced FC size: 256*11*11 -> 512 (was 4096)
            nn.Linear(256 * 11 * 11, 512),
            nn.BatchNorm1d(512), # Added BN for FC
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Reduced FC size: 512 -> 512 (was 4096)
            nn.Linear(512, 512),
            nn.BatchNorm1d(512), # Added BN for FC
            nn.ReLU(),
            nn.Dropout(p=0.5),
            
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def get_model(num_classes=2, dropout=0.5):
    return AlexNet(num_classes=num_classes, dropout=dropout)  # ← 传入 dropout
