import torch
from torch import nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=2):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=5, stride=1, padding=2), # 100 -> 100
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), # 100 -> 49
            
            nn.Conv2d(96, 256, kernel_size=5, padding=2), # 49 -> 49
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), # 49 -> 24
            
            nn.Conv2d(256, 384, kernel_size=3, padding=1), # 24 -> 24
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), # 24 -> 24
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), # 24 -> 24
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), # 24 -> 11
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 11 * 11, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def get_model(num_classes=2):
    return AlexNet(num_classes=num_classes)
