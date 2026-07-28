import torch
from torch import nn

def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=3,
                                padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

class VGG6(nn.Module):
    def __init__(self, num_classes=2):
        super(VGG6, self).__init__()
        conv_arch = ((1, 64), (2, 128))
        conv_blks = []
        in_channels = 3 # RGB
        
        # Convolutional part
        for (num_convs, out_channels) in conv_arch:
            conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
            in_channels = out_channels
            
        self.features = nn.Sequential(*conv_blks)
        
        # Flatten size calculation:
        # 100 -> 50 (after block 1) -> 25 (after block 2)
        # Use AdaptiveAvgPool to reduce size to 5x5
        # 128 * 5 * 5 = 3200
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((5, 5)),
            nn.Flatten(),
            nn.Linear(128 * 5 * 5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def get_model(num_classes=2):
    return VGG6(num_classes=num_classes)