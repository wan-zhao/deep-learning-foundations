import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torchvision.datasets import ImageFolder
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

transforms=transforms.Compose([
    transforms.ToTensor(),#转成tensor类型
    transforms.Normalize(mean=(0.5,),std=(0.5,))#标准化
])
dataset=ImageFolder('mood_train/',transform=transforms)#读取img
#print(dataset[0][0].shape)
dataloader=DataLoader(dataset,batch_size=32,shuffle=True)#小批量数据
class moodcnn(nn.Module):
    def __init__(self):
        super(moodcnn,self).__init__()
        self.conv1=nn.Conv2d(3,32,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(32,64,kernel_size=3,padding=1)
        self.conv3=nn.Conv2d(64,128,kernel_size=3,padding=1)
        self.maxpool=nn.MaxPool2d(kernel_size=2)
        self.fc1=nn.Linear(128*6*6,100)
        self.fc2=nn.Linear(100,7)
    def forward(self,x):
        x=self.maxpool(F.relu(self.conv1(x)))
        x=self.maxpool(F.relu(self.conv2(x)))
        x=self.maxpool(F.relu(self.conv3(x)))
        x=x.view(-1,128*6*6)
        x=F.relu(self.fc1(x))
        x=self.fc2(x)
        return x
model=moodcnn()

if __name__ == '__main__':
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model=model.to(device)
    loss_func=F.cross_entropy
    optimizer = opt.Adam(model.parameters(), lr=0.001)
    epochs=6
    for epoch in range(epochs):
       for i,data in enumerate(dataloader,0):
           inputs,labels=data[0].to(device),data[1].to(device)
           optimizer.zero_grad()
    
           output=model(inputs)
           loss=loss_func(output,labels)
           loss.backward()
           optimizer.step()
    
           if i%10==0:
              print(i,loss)
    print("finished training!")
    print("提升模型方式：1.epochs 2.lr 3.batch_size 4.fc_num 5.fc_参数 6.conv_num ")
    
    # Save the model
    torch.save(model.state_dict(), 'mood_model.pth')
    print("Model saved to mood_model.pth")