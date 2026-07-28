import torch
import torch.nn as nn
import torch.nn.functional as F

class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):
        super(Inception, self).__init__()
        # 线路1: 1x1卷积
        self.p1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 线路2: 1x1卷积 + 3x3卷积
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3: 1x1卷积 + 5x5卷积
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4: 3x3池化 + 1x1卷积
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        o1 = F.relu(self.p1(x))
        o2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        o3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        o4 = F.relu(self.p4_2(self.p4_1(x)))
        
        output = torch.cat((o1, o2, o3, o4), dim=1)
        return output

# 测试实例化
inc = Inception(128, 16, (16, 32), (32, 64), 32)


GoogLeNet_simple = nn.Sequential(
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
    nn.Linear(144, 10)
)

from PIL import Image
import numpy as np

# 假设目录下有 cat.jpg
I = Image.open('cat.jpg')
I = np.array(I)
I = I[::2, ::2]    # 下采样
I = I[:224, :224]  # 裁剪
img = I.reshape(1, 3, 224, 224) 
img = torch.tensor(img, dtype=torch.float32)

# 进行推理
output = GoogLeNet_simple(img)
print(output)