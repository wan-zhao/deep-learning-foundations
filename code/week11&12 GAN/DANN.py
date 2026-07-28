
##~~~~~~~~~~~~~~~~~~~创建MNIST-M数据集~~~~~~~~~~~~~~~~~~##
import torch
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
import requests
import zipfile
import os
import pickle
with open('keras_mnistm.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin1')
train = data['train'] #[60000, 28, 28, 3]
test = data['test']
# MNIST-M数据加载
mnist_transform = transforms.Compose([
    transforms.ToTensor(),
])
class MNISTM(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        super().__init__()
        self.transform = transform
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img = self.data[idx]
        if self.transform:
            img = self.transform(img)
        return img
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 定义模型架构（包含GRL） ~~~~~~~~~~~~~~~~~~##
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
# GRL层定义
class GRL(Function):
    @staticmethod
    def forward(self, x, alpha):
        self.alpha = alpha
        return x
    @staticmethod
    def backward(self, grad_output):
        return grad_output.neg() * self.alpha, None

# 特征提取器
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 48, 5), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(48 * 4 * 4, 100)
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x
# 分类器
class LabelPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(100, 10)
    def forward(self, x):
        return self.fc(x)
# 领域分类器
class DomainClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(100, 100), nn.ReLU(),
            nn.Linear(100, 2)
        )
    def forward(self, x, alpha):
        x = GRL.apply(x, alpha)
        return self.fc(x)
##~~~~~~~~~~~~~~~~~~~~~~~~~ 训练DANN模型~~~~~~~~~~~~~~~~~~~##
import torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 数据加载
source_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.Grayscale(3),
                       # transforms.Resize(32),
                       transforms.ToTensor(),
                   ])),
    batch_size=128, shuffle=True)
target_loader = torch.utils.data.DataLoader(
    MNISTM(train, transform=mnist_transform),
    batch_size=128, shuffle=True)
# 模型实例化
feature_extractor = FeatureExtractor().to(device)
label_predictor = LabelPredictor().to(device)
domain_classifier = DomainClassifier().to(device)
# 优化器
optimizer = optim.Adam([
    {'params': feature_extractor.parameters()},
    {'params': label_predictor.parameters()},
    {'params': domain_classifier.parameters()}
], lr=1e-3)
criterion_label = nn.CrossEntropyLoss()
criterion_domain = nn.CrossEntropyLoss()
# 训练
for epoch in range(10):
    feature_extractor.train()
    label_predictor.train()
    domain_classifier.train()
    len_dataloader = min(len(source_loader), len(target_loader))
    iter_source = iter(source_loader)
    iter_target = iter(target_loader)
    for i in range(len_dataloader):
        p = float(i + epoch * len_dataloader) / (10 * len_dataloader)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1 # 动态调节
        # 源域
        data_source = next(iter_source)
        s_img, s_label = data_source[0].to(device), data_source[1].to(device)
        # 目标域
        # data_target = iter_target.next()
        # t_img, _ = data_target[0].to(device), data_target[1]
        t_img = next(iter_target).to(device)
        optimizer.zero_grad()
        # 前向
        s_feat = feature_extractor(s_img)
        t_feat = feature_extractor(t_img)
        # 分类损失
        class_pred = label_predictor(s_feat)
        loss_class = criterion_label(class_pred, s_label)
        # 域损失
        feat_concat = torch.cat((s_feat, t_feat), 0)
        domain_pred = domain_classifier(feat_concat, alpha)
        domain_labels = torch.cat((
            torch.zeros(s_feat.size(0)).long(),
            torch.ones(t_feat.size(0)).long()
        ), 0).to(device)
        loss_domain = criterion_domain(domain_pred, domain_labels)
        # 总损失
        loss = loss_class + loss_domain
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch+1}/10] loss_class: {loss_class.item():.4f}, loss_domain: {loss_domain.item():.4f}')
