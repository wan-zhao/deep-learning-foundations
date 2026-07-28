import torch
from torch import nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self, num_classes=2):
        super(LeNet5, self).__init__()
        # Input: 3 x 100 x 100
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5) # 100-5+1=96
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 96/2=48
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5) # 48-5+1=44
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 44/2=22
        
        # Flatten size: 16 * 22 * 22 = 7744
        self.fc1 = nn.Linear(16 * 22 * 22, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def get_model(num_classes=2):
    return LeNet5(num_classes=num_classes)