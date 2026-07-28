import torch
from IPython import display
import torchvision
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

lr=0.001
batch_size=128
epochs=2


# 加载数据
train_X,test_X,train_y, test_y =np. load(
    'mnist.npy',allow_pickle=True)
#转换为tensor数据类型
x_train =torch.tensor(train_X, dtype=torch.float32)
y_train =torch.tensor(train_y)
# #转换为dataset
train_ds=TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=batch_size, drop_last=True)

#AlexNet模型搭建，此程序版本图片直接会被压扁，没有改动Alex参数
AlexNet = nn.Sequential(
    nn.Conv2d(1,96,kernel_size=5, stride=1, padding=2), 
    nn. ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),# 输出: (28-3)/2+1=13

    nn.Conv2d(96,256,kernel_size=5, padding=2), 
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),# 输出: (13-3)/2+1=6

    nn.Conv2d(256,384,kernel_size=3, padding=1), 
    nn.ReLU(),
    nn.Conv2d(384,384,kernel_size=3,padding=1),
    nn.ReLU(),
    nn.Conv2d(384,256,kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2), # 输出: (6-3)/2+1=2

    nn. Flatten(), # Flatten: 256 * 2 * 2 = 1024
    nn.Linear(1024, 4096),
    nn.ReLU(),
   
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    #类别数为10
    nn.Linear(4096,10))

import torch.nn.functional as F
loss_func=F.cross_entropy

from torch import optim
opt=optim.SGD(AlexNet.parameters(),lr)

for i in range (epochs):
    j=0
    for xb, yb in train_dl:
        xb=xb.reshape(128,1,28,28)
    # 前向传播
        pred = AlexNet(xb)
    #求损失
        loss = loss_func (pred, yb)
    # 反向传播
        loss. backward ()
    #更参
        opt.step()
        opt.zero_grad()

        print(j)
        j+=1
    print(f"Epoch {epochs+1}, Loss: {loss.item():.4f}")