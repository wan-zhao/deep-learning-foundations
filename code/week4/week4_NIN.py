import torch
import torch.nn as nn
#构造NIN block
def ninblock(in_channels,out_channels,kernel_size,stride,padding):
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding),
        nn.ReLU(),
        nn.Conv2d(out_channels,out_channels,kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(out_channels,out_channels,kernel_size=1),
        nn.ReLU()
    )
#构造NIN网络
ninNet=nn.Sequential(
    ninblock(3,96,kernel_size=11,stride=4,padding=0),
    nn.MaxPool2d(3,stride=2),
    ninblock(96,256,kernel_size=5,stride=1,padding=2),
    nn.MaxPool2d(3,stride=2),
    ninblock(256,384,3,1,1),
    nn.MaxPool2d(3,stride=2),
    ninblock(384,10,3,1,1),
    nn.AdaptiveAvgPool2d((1,1)),
    nn.Flatten()
)

from PIL import Image
import numpy as np
I=Image.open('cat.jpg')
I=np.array(I)
I=I[::2,::2]
I=I[:224,:224]
img=I.reshape(1,3,224,224)
img=torch.tensor(img,dtype=torch.float32)
ninNet(img)

for blk in ninNet:
    X=blk(img)
    print(blk .__class__.__name__,'output shape:\t',X.shape)
    img=torch.randn(X.shape)