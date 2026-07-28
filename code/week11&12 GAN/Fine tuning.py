import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import glob

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

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

print(f"训练集: {len(train_dataset)} 张, 验证集: {len(val_dataset)} 张")

# ============== 加载预训练模型 ==============
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet50(pretrained=True)

# 冻结前面所有层
for param in model.parameters():
    param.requires_grad = False

# 替换分类头 (2分类: 猫/狗)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# 仅训练新的 fc 层
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

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

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

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

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# ============== 开始训练 ==============
num_epochs = 10
best_val_acc = 0

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    print(f'Epoch [{epoch+1}/{num_epochs}]')
    print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_catdog_model.pth')
        print(f'  *** 保存最佳模型 (Acc: {val_acc:.2f}%) ***')
    print()

print(f'训练完成! 最佳验证准确率: {best_val_acc:.2f}%')