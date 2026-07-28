"""
猫狗二分类迁移学习实验 - 比较不同冻结策略
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
from collections import defaultdict
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

        # 从文件名提取标签: cat.x.jpg -> 0, dog.x.jpg -> 1
        label = 0 if filename.lower().startswith('cat') else 1

        if self.transform:
            image = self.transform(image)

        return image, label

# ============== 数据预处理 ==============
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ============== 加载数据 ==============
data_dir = r'C:\TZ\Code\week11&12 GAN\dogs-vs-cats\train\train'
all_files = glob.glob(os.path.join(data_dir, '*.jpg'))

# 划分训练集和验证集 (80% / 20%)
train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)

train_dataset = CatDogDataset(train_files, transform=train_transform)
val_dataset = CatDogDataset(val_files, transform=val_transform)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=(device.type=='cuda'))
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=(device.type=='cuda'))

print(f"训练集: {len(train_dataset)} 张, 验证集: {len(val_dataset)} 张")

# ============== 不同冻结策略 ==============
def get_model_with_freeze_strategy(strategy, num_classes=2):
    """
    strategy: 'fc_only', 'layer4', 'layer3+', 'unfrozen_all'
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet50(pretrained=True)

    # 首先冻结所有参数
    for param in model.parameters():
        param.requires_grad = False

    if strategy == 'fc_only':
        # 只训练fc层
        pass

    elif strategy == 'layer4':
        # 解冻 layer4 和 fc
        for param in model.layer4.parameters():
            param.requires_grad = True

    elif strategy == 'layer3+':
        # 解冻 layer3, layer4 和 fc
        for param in model.layer3.parameters():
            param.requires_grad = True
        for param in model.layer4.parameters():
            param.requires_grad = True

    elif strategy == 'unfrozen_all':
        # 解冻所有层
        for param in model.parameters():
            param.requires_grad = True

    # 替换分类头
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    return model, device

# ============== 优化器配置 ==============
def get_optimizer(model, strategy):
    """根据策略设置不同的学习率"""
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
    else:  # unfrozen_all
        return optim.Adam([
            {'params': model.conv1.parameters(), 'lr': 1e-6},
            {'params': model.bn1.parameters(), 'lr': 1e-6},
            {'params': model.layer1.parameters(), 'lr': 1e-6},
            {'params': model.layer2.parameters(), 'lr': 1e-5},
            {'params': model.layer3.parameters(), 'lr': 1e-5},
            {'params': model.layer4.parameters(), 'lr': 1e-4},
            {'params': model.fc.parameters(), 'lr': 1e-3}
        ])

# ============== 训练和验证函数 ==============
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

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

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

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# ============== 完整训练循环（带历史记录） ==============
def train_experiment(strategy, num_epochs=15):
    print(f"\n{'='*60}")
    print(f"开始训练策略: {strategy}")
    print(f"{'='*60}")

    model, device = get_model_with_freeze_strategy(strategy)
    optimizer = get_optimizer(model, strategy)
    criterion = nn.CrossEntropyLoss()

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    best_val_acc = 0

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'best_model_{strategy}.pth')
            print(f'  *** 保存最佳模型 (Acc: {val_acc:.2f}%) ***')
        print()

    print(f'{strategy} 训练完成! 最佳验证准确率: {best_val_acc:.2f}%')
    return history, best_val_acc

# ============== 绘制对比曲线 ==============
def plot_comparison(all_histories, all_best_accs):
    strategies = list(all_histories.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 训练损失
    for strategy in strategies:
        axes[0, 0].plot(all_histories[strategy]['train_loss'], label=strategy, marker='o', markersize=3)
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 训练精度
    for strategy in strategies:
        axes[0, 1].plot(all_histories[strategy]['train_acc'], label=strategy, marker='o', markersize=3)
    axes[0, 1].set_title('Training Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 验证损失
    for strategy in strategies:
        axes[1, 0].plot(all_histories[strategy]['val_loss'], label=strategy, marker='o', markersize=3)
    axes[1, 0].set_title('Validation Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 验证精度
    for strategy in strategies:
        axes[1, 1].plot(all_histories[strategy]['val_acc'], label=strategy, marker='o', markersize=3)
    axes[1, 1].set_title('Validation Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('catdog_comparison.png', dpi=150)
    print("\n曲线图已保存为: catdog_comparison.png")

    # 打印最佳结果对比
    print("\n" + "="*60)
    print("各策略最佳验证准确率对比:")
    print("="*60)
    for strategy, acc in all_best_accs.items():
        print(f"{strategy:15s}: {acc:.2f}%")
    print("="*60)

# ============== 主程序 ==============
if __name__ == '__main__':
    # 定义要测试的策略
    strategies = ['fc_only', 'layer4', 'layer3+', 'unfrozen_all']

    # 存储所有结果
    all_histories = {}
    all_best_accs = {}

    # 逐个训练
    for strategy in strategies:
        history, best_acc = train_experiment(strategy, num_epochs=15)
        all_histories[strategy] = history
        all_best_accs[strategy] = best_acc

        # 保存结果到JSON
        with open(f'history_{strategy}.json', 'w') as f:
            json.dump({k: [float(x) for x in v] for k, v in history.items()}, f, indent=2)

    # 绘制对比图
    plot_comparison(all_histories, all_best_accs)

    # 保存汇总结果
    summary = {
        'strategies': strategies,
        'best_val_accs': {k: float(v) for k, v in all_best_accs.items()}
    }
    with open('catdog_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n所有实验完成! 结果已保存。")
