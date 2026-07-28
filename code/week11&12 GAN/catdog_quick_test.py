"""
猫狗二分类迁移学习 - 快速测试版 (2000张图片)
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
import json

# ============== 自定义数据集 ==============
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

# ============== 数据预处理 ==============
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ============== 加载数据 (快速测试 - 只用2000张) ==============
data_dir = r'C:\TZ\Code\week11&12 GAN\dogs-vs-cats\train\train'
all_files = glob.glob(os.path.join(data_dir, '*.jpg'))

# 随机采样2000张用于快速测试
np.random.seed(42)
np.random.shuffle(all_files)
sample_files = all_files[:2000]

train_files, val_files = train_test_split(sample_files, test_size=0.2, random_state=42)

train_dataset = CatDogDataset(train_files, transform=train_transform)
val_dataset = CatDogDataset(val_files, transform=val_transform)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

print(f"快速测试模式 - 数据集: {len(sample_files)} 张")
print(f"训练集: {len(train_dataset)} 张, 验证集: {len(val_dataset)} 张")
print(f"设备: {device}")

# ============== 不同冻结策略 ==============
def get_model_with_freeze_strategy(strategy, num_classes=2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    if strategy == 'layer4':
        for param in model.layer4.parameters():
            param.requires_grad = True
    elif strategy == 'layer3+':
        for param in model.layer3.parameters():
            param.requires_grad = True
        for param in model.layer4.parameters():
            param.requires_grad = True
    elif strategy == 'unfrozen_all':
        for param in model.parameters():
            param.requires_grad = True

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    return model, device

def get_optimizer(model, strategy):
    if strategy == 'fc_only':
        return optim.Adam(model.fc.parameters(), lr=1e-3)
    elif strategy == 'layer4':
        return optim.Adam([
            {'params': model.layer4.parameters(), 'lr': 1e-4},
            {'params': model.fc.parameters(), 'lr': 1e-3}
        ])
    elif strategy == 'layer3+':
        return optim.Adam([
            {'params': model.layer3.parameters(), 'lr': 1e-5},
            {'params': model.layer4.parameters(), 'lr': 1e-4},
            {'params': model.fc.parameters(), 'lr': 1e-3}
        ])
    else:
        return optim.Adam([
            {'params': model.conv1.parameters(), 'lr': 1e-6},
            {'params': model.layer1.parameters(), 'lr': 1e-6},
            {'params': model.layer2.parameters(), 'lr': 1e-5},
            {'params': model.layer3.parameters(), 'lr': 1e-5},
            {'params': model.layer4.parameters(), 'lr': 1e-4},
            {'params': model.fc.parameters(), 'lr': 1e-3}
        ])

# ============== 训练和验证 ==============
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

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
    running_loss = 0.0
    correct = 0
    total = 0

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

# ============== 训练实验 ==============
def train_experiment(strategy, num_epochs=5):
    print(f"\n{'='*60}")
    print(f"策略: {strategy} | Epochs: {num_epochs}")
    print(f"{'='*60}")

    model, device = get_model_with_freeze_strategy(strategy)
    optimizer = get_optimizer(model, strategy)
    criterion = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f'Epoch [{epoch+1}/{num_epochs}] Train: {train_acc:.1f}% Val: {val_acc:.1f}%')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'quick_{strategy}.pth')

    print(f'最佳验证准确率: {best_val_acc:.2f}%')
    return history, best_val_acc

# ============== 绘图 ==============
def plot_comparison(all_histories):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for strategy, history in all_histories.items():
        axes[0, 0].plot(history['train_loss'], label=strategy, marker='o')
        axes[0, 1].plot(history['train_acc'], label=strategy, marker='o')
        axes[1, 0].plot(history['val_loss'], label=strategy, marker='o')
        axes[1, 1].plot(history['val_acc'], label=strategy, marker='o')

    axes[0, 0].set_title('Training Loss')
    axes[0, 1].set_title('Training Accuracy')
    axes[1, 0].set_title('Validation Loss')
    axes[1, 1].set_title('Validation Accuracy')

    for ax in axes.flat:
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Epoch')

    plt.tight_layout()
    plt.savefig('quick_comparison.png')
    print("\n曲线图已保存: quick_comparison.png")

# ============== 主程序 ==============
if __name__ == '__main__':
    torch.manual_seed(42)

    strategies = ['fc_only', 'layer4', 'layer3+']
    all_histories = {}
    all_best_accs = {}

    for strategy in strategies:
        history, best_acc = train_experiment(strategy, num_epochs=5)
        all_histories[strategy] = history
        all_best_accs[strategy] = best_acc

    plot_comparison(all_histories)

    print("\n" + "="*60)
    print("结果汇总:")
    for strategy, acc in all_best_accs.items():
        print(f"  {strategy}: {acc:.2f}%")
    print("="*60)
