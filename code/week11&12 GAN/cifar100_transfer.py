"""
CIFAR-100 迁移学习实验
使用预训练ResNet50进行微调
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import json
from collections import defaultdict
import time

# ============== 数据增强策略 ==============
def get_data_transforms():
    """
    CIFAR-100 数据增强
    - 训练集: 随机裁剪、翻转、颜色增强
    - 验证集: 标准化处理
    """
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform

# ============== 加载CIFAR-100数据集 ==============
def load_cifar100(data_path='./data', batch_size=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_transform, val_transform = get_data_transforms()

    # 加载训练集并划分验证集
    train_dataset = datasets.CIFAR100(root=data_path, train=True, download=True, transform=train_transform)
    val_dataset = datasets.CIFAR100(root=data_path, train=True, download=True, transform=val_transform)

    # 手动划分训练集和验证集 (90% / 10%)
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, _ = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    _, val_dataset = torch.utils.data.random_split(val_dataset, [train_size, val_size])

    # 测试集
    test_dataset = datasets.CIFAR100(root=data_path, train=False, download=True, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=(device.type=='cuda'))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=(device.type=='cuda'))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=(device.type=='cuda'))

    print(f"CIFAR-100 数据加载完成:")
    print(f"  训练集: {len(train_dataset)} 张")
    print(f"  验证集: {len(val_dataset)} 张")
    print(f"  测试集: {len(test_dataset)} 张")

    return train_loader, val_loader, test_loader

# ============== 冻结策略配置 ==============
def apply_freeze_strategy(model, strategy):
    """
    冻结策略:
    - 'fc_only': 只训练分类头 (100类)
    - 'layer4': 解冻 layer4 + fc
    - 'layer3+': 解冻 layer3 + layer4 + fc
    - 'unfrozen': 全部解冻
    """
    # 先冻结所有层
    for param in model.parameters():
        param.requires_grad = False

    if strategy == 'fc_only':
        pass  # 保持全部冻结，只训练fc
    elif strategy == 'layer4':
        for param in model.layer4.parameters():
            param.requires_grad = True
    elif strategy == 'layer3+':
        for param in model.layer3.parameters():
            param.requires_grad = True
        for param in model.layer4.parameters():
            param.requires_grad = True
    elif strategy == 'unfrozen':
        for param in model.parameters():
            param.requires_grad = True

    return model

# ============== 学习率设置策略 ==============
def get_optimizer_and_scheduler(model, strategy, epochs):
    """
    分层学习率策略 + 余弦退火调度器
    """
    if strategy == 'fc_only':
        optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
    elif strategy == 'layer4':
        optimizer = optim.Adam([
            {'params': model.layer4.parameters(), 'lr': 1e-4},
            {'params': model.fc.parameters(), 'lr': 1e-3}
        ])
    elif strategy == 'layer3+':
        optimizer = optim.Adam([
            {'params': model.layer3.parameters(), 'lr': 5e-5},
            {'params': model.layer4.parameters(), 'lr': 1e-4},
            {'params': model.fc.parameters(), 'lr': 1e-3}
        ])
    else:  # unfrozen
        optimizer = optim.Adam([
            {'params': model.conv1.parameters(), 'lr': 1e-6},
            {'params': model.bn1.parameters(), 'lr': 1e-6},
            {'params': model.layer1.parameters(), 'lr': 1e-6},
            {'params': model.layer2.parameters(), 'lr': 1e-5},
            {'params': model.layer3.parameters(), 'lr': 1e-5},
            {'params': model.layer4.parameters(), 'lr': 1e-4},
            {'params': model.fc.parameters(), 'lr': 1e-3}
        ])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    return optimizer, scheduler

# ============== 创建模型 ==============
def create_model(strategy):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet50(pretrained=True)

    # 应用冻结策略
    model = apply_freeze_strategy(model, strategy)

    # 替换分类头 (100类)
    model.fc = nn.Linear(model.fc.in_features, 100)
    model = model.to(device)

    return model, device

# ============== 训练函数 ==============
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

# ============== 验证函数 ==============
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

# ============== 完整训练流程 ==============
def train_cifar100(strategy='fc_only', epochs=30, batch_size=128):
    print(f"\n{'='*70}")
    print(f"CIFAR-100 迁移学习 - 策略: {strategy}")
    print(f"{'='*70}")

    # 加载数据
    train_loader, val_loader, test_loader = load_cifar100(batch_size=batch_size)

    # 创建模型
    model, device = create_model(strategy)
    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = get_optimizer_and_scheduler(model, strategy, epochs)

    # 记录历史
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': []
    }

    best_val_acc = 0
    start_time = time.time()

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        print(f'Epoch [{epoch+1}/{epochs}] LR: {current_lr:.6f}')
        print(f'  Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
        print(f'  Val   Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'cifar100_best_{strategy}.pth')
            print(f'  *** Best model saved! (Val Acc: {val_acc:.2f}%) ***')

    # 测试集评估
    model.load_state_dict(torch.load(f'cifar100_best_{strategy}.pth'))
    test_loss, test_acc = validate(model, test_loader, criterion, device)

    elapsed_time = time.time() - start_time
    print(f"\n{strategy} 训练完成!")
    print(f"  最佳验证准确率: {best_val_acc:.2f}%")
    print(f"  测试集准确率: {test_acc:.2f}%")
    print(f"  训练时间: {elapsed_time/60:.1f} 分钟")

    return history, best_val_acc, test_acc

# ============== 绘制训练曲线 ==============
def plot_training_curves(history, strategy):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 损失曲线
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o', markersize=3)
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s', markersize=3)
    axes[0].set_title(f'{strategy} - Loss Curve')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 精度曲线
    axes[1].plot(history['train_acc'], label='Train Acc', marker='o', markersize=3)
    axes[1].plot(history['val_acc'], label='Val Acc', marker='s', markersize=3)
    axes[1].set_title(f'{strategy} - Accuracy Curve')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 学习率曲线
    axes[2].plot(history['lr'], label='Learning Rate', marker='o', markersize=3, color='green')
    axes[2].set_title(f'{strategy} - Learning Rate Schedule')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_yscale('log')

    plt.tight_layout()
    plt.savefig(f'cifar100_curves_{strategy}.png', dpi=150)
    print(f"训练曲线已保存: cifar100_curves_{strategy}.png")

# ============== 多策略对比 ==============
def compare_strategies(strategies=['fc_only', 'layer4', 'layer3+'], epochs=30):
    all_results = {}

    for strategy in strategies:
        history, best_val, test_acc = train_cifar100(strategy, epochs=epochs)
        all_results[strategy] = {
            'history': history,
            'best_val_acc': best_val,
            'test_acc': test_acc
        }

        # 保存历史
        with open(f'cifar100_history_{strategy}.json', 'w') as f:
            json.dump({k: [float(x) if isinstance(x, (int, float)) else x for k, v in history.items()]
                      for k, v in history.items()}, f, indent=2)

        # 绘制单独曲线
        plot_training_curves(history, strategy)

    # 绘制对比图
    plot_comparison(all_results)

    # 打印总结
    print("\n" + "="*70)
    print("各策略性能对比:")
    print("="*70)
    print(f"{'策略':<15} {'最佳验证准确率':<15} {'测试集准确率':<15}")
    print("-"*70)
    for strategy, results in all_results.items():
        print(f"{strategy:<15} {results['best_val_acc']:<15.2f} {results['test_acc']:<15.2f}")
    print("="*70)

    # 保存汇总
    summary = {k: {'best_val_acc': float(v['best_val_acc']), 'test_acc': float(v['test_acc'])}
               for k, v in all_results.items()}
    with open('cifar100_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    return all_results

# ============== 策略对比图 ==============
def plot_comparison(all_results):
    strategies = list(all_results.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 验证精度对比
    for strategy in strategies:
        axes[0].plot(all_results[strategy]['history']['val_acc'],
                    label=strategy, marker='o', markersize=4)
    axes[0].set_title('Validation Accuracy Comparison')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 测试准确率柱状图
    test_accs = [all_results[s]['test_acc'] for s in strategies]
    bars = axes[1].bar(strategies, test_accs, color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(strategies)])
    axes[1].set_title('Test Accuracy Comparison')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_ylim(min(test_accs) - 2, max(test_accs) + 2)

    for bar, acc in zip(bars, test_accs):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{acc:.2f}%', ha='center', fontsize=11)

    plt.tight_layout()
    plt.savefig('cifar100_comparison.png', dpi=150)
    print("\n对比图已保存: cifar100_comparison.png")

# ============== 主程序 ==============
if __name__ == '__main__':
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    print("="*70)
    print("CIFAR-100 迁移学习实验")
    print("="*70)
    print("\n实验设置:")
    print("  - 模型: ResNet50 (预训练于 ImageNet)")
    print("  - 数据集: CIFAR-100 (100类, 45k训练, 5k测试)")
    print("  - 数据增强: 随机裁剪、翻转、旋转、颜色抖动")
    print("  - 学习率调度: Cosine Annealing")
    print("  - 冻结策略: fc_only / layer4 / layer3+")
    print("="*70)

    # 运行对比实验
    results = compare_strategies(
        strategies=['fc_only', 'layer4', 'layer3+'],
        epochs=30
    )

    print("\n实验完成! 所有结果已保存。")
