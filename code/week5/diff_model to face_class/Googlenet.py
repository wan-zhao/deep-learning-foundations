import torch
import torch.nn as nn
import torch.nn.functional as F

class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):
        super(Inception, self).__init__()
        # Path 1: 1x1 conv
        self.p1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # Path 2: 1x1 conv + 3x3 conv
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # Path 3: 1x1 conv + 5x5 conv
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # Path 4: 3x3 pool + 1x1 conv
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        o1 = F.relu(self.p1(x))
        o2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        o3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        o4 = F.relu(self.p4_2(self.p4_1(x)))
        
        output = torch.cat((o1, o2, o3, o4), dim=1)
        return output

class GoogLeNet(nn.Module):
    def __init__(self, num_classes=2):
        super(GoogLeNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(32, 64, kernel_size=1),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            Inception(128, 16, (16, 32), (32, 64), 32),
            Inception(144, 16, (16, 32), (32, 64), 32),
            Inception(144, 16, (16, 32), (32, 64), 32),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(144, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def get_model(num_classes=2):
    return GoogLeNet(num_classes=num_classes)