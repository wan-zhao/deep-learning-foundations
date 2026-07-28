import torch
from torch import nn

def ninblock(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU()
    )

class NIN(nn.Module):
    def __init__(self, num_classes=2):
        super(NIN, self).__init__()
        self.net = nn.Sequential(
            ninblock(3, 96, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(3, stride=2),
            ninblock(96, 256, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(3, stride=2),
            ninblock(256, 384, 3, 1, 1),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout(0.5),
            ninblock(384, num_classes, 3, 1, 1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def forward(self, x):
        return self.net(x)

def get_model(num_classes=2):
    return NIN(num_classes=num_classes)