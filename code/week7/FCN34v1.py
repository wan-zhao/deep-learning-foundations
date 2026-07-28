import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import random

# 定义 Pascal VOC 数据集
class VOCSegmentationDataset(Dataset):
    def __init__(self, root, image_set='train', transforms=None):
        self.root = root
        self.transforms = transforms
        self.image_set = image_set 
        image_dir = os.path.join(root, "JPEGImages")
        mask_dir = os.path.join(root, "SegmentationClass")
        split_file = os.path.join(root, "ImageSets/Segmentation", f"{image_set}.txt")
        
        with open(split_file, "r") as file:
            file_names = file.read().splitlines()
            
        self.images = [os.path.join(image_dir, f"{x}.jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, f"{x}.png") for x in file_names]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        mask = Image.open(self.masks[idx])
    
        # 1. 随机调整大小 (Scale Jittering)
        if self.image_set == 'train':
            resize_scale = random.uniform(0.5, 2.0)
            target_size = int(256 * resize_scale)
            image = image.resize((target_size, target_size), Image.BILINEAR)
            mask = mask.resize((target_size, target_size), Image.NEAREST)
            
            # 训练时：随机裁剪到固定尺寸 256x256
            if target_size > 256:
                # 如果缩放后的尺寸大于256，进行随机裁剪
                i = random.randint(0, target_size - 256)
                j = random.randint(0, target_size - 256)
                image = TF.crop(image, i, j, 256, 256)
                mask = TF.crop(mask, i, j, 256, 256)
            else:
                # 如果缩放后的尺寸小于256，先padding再裁剪
                padding = (256 - target_size) // 2
                image = TF.pad(image, padding, fill=0, padding_mode='constant')
                mask = TF.pad(mask, padding, fill=255, padding_mode='constant')
                # 确保是256x256
                image = TF.center_crop(image, 256)
                mask = TF.center_crop(mask, 256)
        else:
            # 验证时：直接resize到256x256
            image = image.resize((256, 256), Image.BILINEAR)
            mask = mask.resize((256, 256), Image.NEAREST)

        # 2. 随机水平翻转 (Random Horizontal Flip)
        if self.image_set == 'train' and random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
            
        # 3. 转 Tensor 和 归一化
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        mask[mask > 20] = 255
        
        return image, mask

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集
# 注意：你需要将 root 修改为你本地的实际路径
root = r'C:\Users\Lenovo\Documents\python_code\2025ML\week7\VOCdevkit\VOC2007' 
train_dataset = VOCSegmentationDataset(root, image_set='train', transforms=transform)
val_dataset = VOCSegmentationDataset(root, image_set='val', transforms=transform)

def collate_fn(batch):
    images, masks = zip(*batch)
    images = torch.stack(images) # 堆叠图像
    masks = torch.stack(masks)   # 堆叠标签
    return images, masks

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_fn)
# 定义 FCN 网络，使用 ResNet18 (代码中实际写的是 resnet34) 作为编码器

class FCN_ResNet34(nn.Module):
    def __init__(self, num_classes):
        super(FCN_ResNet34, self).__init__()
        # 加载预训练的 ResNet34
        resnet = models.resnet34(pretrained=True)
        
        # 拆分 ResNet 的各层，以便获取中间特征
        self.initial = nn.Sequential(*list(resnet.children())[:4]) # output: 64通道, 1/4 尺寸
        self.layer1 = resnet.layer1 # output: 64通道, 1/4 尺寸
        self.layer2 = resnet.layer2 # output: 128通道, 1/8 尺寸
        self.layer3 = resnet.layer3 # output: 256通道, 1/16 尺寸
        self.layer4 = resnet.layer4 # output: 512通道, 1/32 尺寸
        
        # 定义由于融合特征后的降维层或平滑层
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2) 
        self.up3 = nn.ConvTranspose2d(256 + 256, 128, 2, stride=2) # 256来自up4, 256来自layer3
        self.up2 = nn.ConvTranspose2d(128 + 128, 64, 2, stride=2)  # 128来自up3, 128来自layer2
        
        # 最后的分类层
        self.final_conv = nn.Conv2d(64 + 64, num_classes, kernel_size=1) # 64来自up2, 64来自layer1
        
    def forward(self, x):
        # 编码路径 (Encoder)
        x0 = self.initial(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        # 解码路径 (Decoder) with Skip Connections
        # 1. 上采样 x4 并与 x3 拼接
        u4 = self.up4(x4)
        # 注意：如果尺寸因padding不匹配，需要interpolate对齐，这里简化处理
        if u4.size() != x3.size():
            u4 = F.interpolate(u4, size=x3.shape[2:], mode='bilinear', align_corners=True)
        cat3 = torch.cat((u4, x3), dim=1)
        
        # 2. 上采样 并与 x2 拼接
        u3 = self.up3(cat3)
        if u3.size() != x2.size():
            u3 = F.interpolate(u3, size=x2.shape[2:], mode='bilinear', align_corners=True)
        cat2 = torch.cat((u3, x2), dim=1)
        
        # 3. 上采样 并与 x1 拼接
        u2 = self.up2(cat2)
        if u2.size() != x1.size():
            u2 = F.interpolate(u2, size=x1.shape[2:], mode='bilinear', align_corners=True)
        cat1 = torch.cat((u2, x1), dim=1)
        
        # 4. 最终分类
        out = self.final_conv(cat1)
        # 最后上采样回原图尺寸 (x4倍)
        return F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=True)
# 模型训练
def train_fcn(model, dataloader, epochs, lr, device):
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    
    # 使用 LambdaLR 实现 PolyLR（多项式衰减）
    lambda_lr = lambda epoch: (1 - epoch / epochs) ** 0.9
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
    
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            outputs = F.interpolate(outputs, size=masks.shape[1:], mode='bilinear', align_corners=True)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        # 在每个 epoch 结束时更新学习率
        scheduler.step()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")
# 训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 21 # Pascal VOC 的类别数
model = FCN_ResNet34(num_classes)
train_fcn(model, train_loader, epochs=20, lr=0.01, device=device)

# 可视化结果
def visualize_prediction(model, dataset, device, sample_indices=[0, 10, 20, 30]):
    """
    可视化多个样本的预测结果
    
    Args:
        model: 训练好的模型
        dataset: 数据集
        device: 设备
        sample_indices: 要可视化的样本索引列表（默认为不连续的4个样本）
    """
    model.eval()
    import matplotlib.colors as mcolors
    # 创建一个自定义的colormap，将255映射为黑色，其他类别用不同颜色
    cmap = plt.cm.get_cmap('tab20', 21)  # 21个类别
    
    num_samples = len(sample_indices)
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    
    # 如果只有一个样本，axes需要变成二维数组
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for row_idx, sample_idx in enumerate(sample_indices):
        # 确保索引不超出数据集范围
        if sample_idx >= len(dataset):
            print(f"警告：索引 {sample_idx} 超出数据集范围，跳过")
            continue
            
        # 获取样本
        image, mask = dataset[sample_idx]
        # 将图像转移到设备上并增加 batch 维度
        image_tensor = image.to(device).unsqueeze(0)
        
        with torch.no_grad():
            output = model(image_tensor)
            # 插值调整输出尺寸
            output = F.interpolate(output, size=mask.shape, mode='bilinear', align_corners=True)
            # 获取预测结果
            pred = output.argmax(dim=1).squeeze().cpu().numpy()
        
        mask = mask.cpu().numpy()
        
        # 【调试信息】打印预测结果的统计
        print(f"\n样本 {sample_idx}:")
        print(f"  预测结果的唯一值: {np.unique(pred)}")
        print(f"  预测结果中各类别的像素数量:")
        for val in np.unique(pred):
            count = np.sum(pred == val)
            print(f"    类别 {val}: {count} 像素 ({100*count/pred.size:.2f}%)")
        print(f"  Ground Truth的唯一值: {np.unique(mask)}")
        
        # 反归一化图像以便正确显示颜色
        image_np = image.permute(1, 2, 0).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = std * image_np + mean
        image_np = np.clip(image_np, 0, 1)

        # 显示原图
        axes[row_idx, 0].imshow(image_np)
        axes[row_idx, 0].set_title(f'Image (Sample {sample_idx})')
        axes[row_idx, 0].axis('off')
        
        # 显示 Ground Truth
        axes[row_idx, 1].imshow(mask, cmap=cmap, vmin=0, vmax=20, interpolation='nearest')
        axes[row_idx, 1].set_title(f'Ground Truth (Sample {sample_idx})')
        axes[row_idx, 1].axis('off')
        
        # 显示预测结果
        axes[row_idx, 2].imshow(pred, cmap=cmap, vmin=0, vmax=20, interpolation='nearest')
        axes[row_idx, 2].set_title(f'Prediction (Sample {sample_idx})')
        axes[row_idx, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

# 调用可视化函数，显示4个不连续的样本
if __name__ == '__main__':
    # 可以自定义要显示的样本索引，这里选择 0, 10, 20, 30
    visualize_prediction(model, val_dataset, device, sample_indices=[0, 10, 20, 30])