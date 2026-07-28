import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn

lr = 0.001
batch_size = 128
epochs = 16

# 假设当前目录下有 mnist.npy 文件
train_X, test_X, train_y, test_y = np.load(
    'mnist.npy', allow_pickle=True)

x_train = torch.tensor(train_X, dtype=torch.float32)
y_train = torch.tensor(train_y)
train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=batch_size, drop_last=True)

# 定义卷积层架构配置
conv_arch_2 = ((1, 64), (2, 128))

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


def vgg(conv_arch, in_channels=1):
    conv_blks = []
    in_channels = in_channels
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
        
    return nn.Sequential(
        *conv_blks,
        nn.Flatten(),
        # 全连接层部分
        # 这里的 7*7 是基于 MNIST (28x28) 经过若干次池化后的尺寸，需根据实际情况调整
        nn.Linear(out_channels * 7 * 7, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 10))

net_6 = vgg(conv_arch_2, in_channels=1)

import torch.nn.functional as F
loss_func = F.cross_entropy

from torch import optim
opt = optim.SGD(net_6.parameters(), lr)

for i in range(epochs):
    j = 0
    for xb, yb in train_dl:
        xb = xb.reshape(128, 1, 28, 28)
        
        # 前向传播
        pred = net_6(xb)
        
        # 求损失
        loss = loss_func(pred, yb)
        
        # 反向传播
        loss.backward()
        
        # 更参
        opt.step()
        opt.zero_grad()
        
        print(j)
        j += 1
    print(i, loss)


# 用于检查每一层输出的维度是否正确
# 假设 xb 已经在上面的循环中被定义
for blk in net_6:
    X = blk(xb)
    print(blk.__class__.__name__, 'output shape:\t', X.shape)
    # 这里生成随机数据传入下一层以测试形状匹配
    xb = torch.randn(X.shape)