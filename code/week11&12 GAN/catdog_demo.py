"""
猫狗二分类 - 超快速演示版 (5分钟内完成)
使用ResNet18 + 500张图片
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import os
import glob
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

class CatDogDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform
    def __len__(self):
        return len(self.file_paths)
    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert('RGB')
        filename = os.path.basename(img_path)
        label = 0 if filename.lower().startswith('cat') else 1
        if self.transform:
            image = self.transform(image)
        return image, label

# 数据预处理 (简化版)
train_transform = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 只用500张图片演示
data_dir = r'C:\TZ\Code\week11&12 GAN\dogs-vs-cats\train\train'
all_files = glob.glob(os.path.join(data_dir, '*.jpg'))
np.random.seed(42)
np.random.shuffle(all_files)
sample_files = all_files[:500]  # 只用500张

train_files, val_files = train_test_split(sample_files, test_size=0.2, random_state=42)
train_dataset = CatDogDataset(train_files, transform=train_transform)
val_dataset = CatDogDataset(val_files, transform=val_transform)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

print(f"演示模式 - 数据集: {len(sample_files)} 张")
print(f"训练集: {len(train_dataset)} 张, 验证集: {len(val_dataset)} 张")
print(f"设备: {device}")

# 使用ResNet18 (比ResNet50快很多)
def get_model(strategy):
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    if strategy == 'layer4':
        for param in model.layer4.parameters():
            param.requires_grad = True
    elif strategy == 'unfrozen':
        for param in model.parameters():
            param.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model.to(device)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return running_loss / len(loader), 100. * correct / total

# 训练三种策略
strategies = ['fc_only', 'layer4', 'unfrozen']
results = {}

for strategy in strategies:
    print(f"\n{'='*50}")
    print(f"策略: {strategy}")
    print(f"{'='*50}")

    model = get_model(strategy)
    if strategy == 'fc_only':
        optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
    elif strategy == 'layer4':
        optimizer = optim.Adam([
            {'params': model.layer4.parameters(), 'lr': 1e-4},
            {'params': model.fc.parameters(), 'lr': 1e-3}
        ])
    else:
        optimizer = optim.Adam([
            {'params': model.layer1.parameters(), 'lr': 1e-5},
            {'params': model.layer2.parameters(), 'lr': 1e-5},
            {'params': model.layer3.parameters(), 'lr': 1e-4},
            {'params': model.layer4.parameters(), 'lr': 1e-4},
            {'params': model.fc.parameters(), 'lr': 1e-3}
        ])

    criterion = nn.CrossEntropyLoss()
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0

    for epoch in range(5):  # 只训练5个epoch
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        print(f'Epoch [{epoch+1}/5] Train: {train_acc:.1f}% Val: {val_acc:.1f}%')
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'demo_{strategy}.pth')

    results[strategy] = history
    print(f'最佳验证准确率: {best_val_acc:.2f}%')

# 绘图
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for strategy, history in results.items():
    axes[0, 0].plot(history['train_loss'], label=strategy, marker='o')
    axes[0, 1].plot(history['train_acc'], label=strategy, marker='o')
    axes[1, 0].plot(history['val_loss'], label=strategy, marker='o')
    axes[1, 1].plot(history['val_acc'], label=strategy, marker='o')

axes[0, 0].set_title('Training Loss')
axes[0, 1].set_title('Training Accuracy')
axes[1, 0].set_title('Validation Loss')
axes[1, 1].set_title('Validation Accuracy')
for ax in axes.flat:
    ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('demo_comparison.png', dpi=150)
print("\n曲线图已保存: demo_comparison.png")

print("\n" + "="*50)
print("演示完成! 结果汇总:")
for strategy, history in results.items():
    print(f"  {strategy}: {max(history['val_acc']):.2f}%")
print("="*50)
