import torch
import torch.nn as nn
import torch.nn.functional as F

#定义残差块 (Residual Block)
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, 
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return F.relu(y)

# 测试代码片段
blk = Residual(3, 3)
X = torch.rand(4, 3, 6, 6)
Y = blk(X)
print(Y.shape) # 对应图中输出 torch.Size([4, 3, 6, 6])

blk = Residual(3, 6, use_1x1conv=True, strides=2)
print(blk(X).shape) # 对应图中输出 torch.Size([4, 6, 3, 3])

#定义网络主体部分
def resnet_block(input_channels, num_channels, num_residuals, 
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

#实现网络结构与数据预处理
b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(512, 10))

# 图片读取与处理部分
from PIL import Image
import numpy as np

# 假设当前目录下有一张名为 cat.jpg 的图片
I = Image.open('cat.jpg') 
I = np.array(I)
I = I[::2, ::2]
I = I[:224, :224]
img = I.reshape(1, 3, 224, 224) 
img = torch.tensor(img, dtype=torch.float32)
print(net(img))

for blk in net:
    X=blk(img)
    print(blk .__class__.__name__,'output shape:\t',X.shape)
    img=torch.randn(X.shape)