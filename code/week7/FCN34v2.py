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

# ==========================================
# 1. 数据集定义 (保持不变)
# ==========================================
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
                i = random.randint(0, target_size - 256)
                j = random.randint(0, target_size - 256)
                image = TF.crop(image, i, j, 256, 256)
                mask = TF.crop(mask, i, j, 256, 256)
            else:
                padding = (256 - target_size) // 2
                image = TF.pad(image, padding, fill=0, padding_mode='constant')
                mask = TF.pad(mask, padding, fill=255, padding_mode='constant')
                image = TF.center_crop(image, 256)
                mask = TF.center_crop(mask, 256)
        else:
            # 验证时：直接resize到256x256
            image = image.resize((256, 256), Image.BILINEAR)
            mask = mask.resize((256, 256), Image.NEAREST)

        # 2. 随机水平翻转
        if self.image_set == 'train' and random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
            
        # 3. 转 Tensor 和 归一化
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        mask[mask > 20] = 255
        
        return image, mask

# ==========================================
# 2. 损失函数定义 (新增 DiceLoss)
# ==========================================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1, ignore_index=255):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs: (N, C, H, W) -> 已经是 softmax 后的概率
        # targets: (N, H, W)
        
        num_classes = inputs.shape[1]
        
        # 创建掩码，忽略 255 的区域
        mask = (targets != self.ignore_index)
        
        # 将 targets 中 255 的位置临时变为 0，防止 one_hot 报错（之后会被 mask 掉）
        targets_masked = targets.clone()
        targets_masked[~mask] = 0
        
        # 转为 One-hot 编码: (N, C, H, W)
        true_1_hot = torch.eye(num_classes, device=inputs.device)[targets_masked]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        
        # 应用掩码：只计算有效区域
        inputs = inputs * mask.unsqueeze(1)
        true_1_hot = true_1_hot * mask.unsqueeze(1)
        
        # 计算 Intersection 和 Union
        dims = (0, 2, 3)
        intersection = torch.sum(inputs * true_1_hot, dims)
        cardinality = torch.sum(inputs + true_1_hot, dims)
        
        # 计算 Dice 系数
        dice = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        
        # 返回 1 - mean_dice
        # 我们可以选择跳过背景类 (idx 0) 的 Dice，也可以取平均。
        # 这里取平均，但背景类已经被 CrossEntropy 降权了，所以影响可控
        return 1 - dice.mean()

# ==========================================
# 3. 网络结构 (FCN + ResNet34 + Skip)
# ==========================================
class FCN_ResNet34(nn.Module):
    def __init__(self, num_classes):
        super(FCN_ResNet34, self).__init__()
        resnet = models.resnet34(pretrained=True)
        self.initial = nn.Sequential(*list(resnet.children())[:4]) 
        self.layer1 = resnet.layer1 
        self.layer2 = resnet.layer2 
        self.layer3 = resnet.layer3 
        self.layer4 = resnet.layer4 
        
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2) 
        self.up3 = nn.ConvTranspose2d(256 + 256, 128, 2, stride=2) 
        self.up2 = nn.ConvTranspose2d(128 + 128, 64, 2, stride=2)  
        
        self.final_conv = nn.Conv2d(64 + 64, num_classes, kernel_size=1) 
        
    def forward(self, x):
        x0 = self.initial(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        u4 = self.up4(x4)
        if u4.size() != x3.size():
            u4 = F.interpolate(u4, size=x3.shape[2:], mode='bilinear', align_corners=True)
        cat3 = torch.cat((u4, x3), dim=1)
        
        u3 = self.up3(cat3)
        if u3.size() != x2.size():
            u3 = F.interpolate(u3, size=x2.shape[2:], mode='bilinear', align_corners=True)
        cat2 = torch.cat((u3, x2), dim=1)
        
        u2 = self.up2(cat2)
        if u2.size() != x1.size():
            u2 = F.interpolate(u2, size=x1.shape[2:], mode='bilinear', align_corners=True)
        cat1 = torch.cat((u2, x1), dim=1)
        
        out = self.final_conv(cat1)
        return F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=True)

# ==========================================
# 4. 训练流程 (包含 Loss 优化和 Epoch 调整)
# ==========================================
def train_fcn(model, dataloader, epochs, lr, device):
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    
    # PolyLR 策略
    lambda_lr = lambda epoch: (1 - epoch / epochs) ** 0.9
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
    
    # 【重点修改 1】设置类别权重
    # 背景(0)权重设为0.1，其他物体设为1.0
    weights = torch.ones(21).to(device)
    weights[0] = 0.1 
    
    # CrossEntropy Loss (带权重)
    criterion_ce = nn.CrossEntropyLoss(weight=weights, ignore_index=255)
    
    # 【重点修改 2】Dice Loss
    criterion_dice = DiceLoss(ignore_index=255)

    print(f"开始训练... 总轮数: {epochs}")
    print("使用优化策略: CrossEntropy(Background Weight=0.1) + DiceLoss")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        ce_loss_val = 0
        dice_loss_val = 0
        
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            
            # 前向传播
            outputs = model(images)
            # 确保输出尺寸和 Mask 一致
            outputs = F.interpolate(outputs, size=masks.shape[1:], mode='bilinear', align_corners=True)
            
            # 计算 CrossEntropy Loss
            loss_ce = criterion_ce(outputs, masks)
            
            # 计算 Dice Loss (需要先 Softmax)
            outputs_soft = F.softmax(outputs, dim=1)
            loss_dice = criterion_dice(outputs_soft, masks)
            
            # 【重点修改 3】组合 Loss
            loss = loss_ce + loss_dice 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            ce_loss_val += loss_ce.item()
            dice_loss_val += loss_dice.item()
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # 打印详细 Loss 信息
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Total Loss: {epoch_loss/len(dataloader):.4f} "
              f"(CE: {ce_loss_val/len(dataloader):.4f}, Dice: {dice_loss_val/len(dataloader):.4f}) "
              f"LR: {current_lr:.2e}")

# ==========================================
# 5. 可视化工具
# ==========================================
def visualize_prediction(model, dataset, device, sample_indices=[0, 10, 20, 30]):
    model.eval()
    cmap = plt.cm.get_cmap('tab20', 21)
    
    num_samples = len(sample_indices)
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1: axes = axes.reshape(1, -1)
    
    for row_idx, sample_idx in enumerate(sample_indices):
        if sample_idx >= len(dataset): continue
        image, mask = dataset[sample_idx]
        image_tensor = image.to(device).unsqueeze(0)
        
        with torch.no_grad():
            output = model(image_tensor)
            output = F.interpolate(output, size=mask.shape, mode='bilinear', align_corners=True)
            pred = output.argmax(dim=1).squeeze().cpu().numpy()
        
        mask = mask.cpu().numpy()
        
        print(f"\n样本 {sample_idx}: GT类别: {np.unique(mask)} -> 预测类别: {np.unique(pred)}")
        
        # 反归一化
        image_np = image.permute(1, 2, 0).cpu().numpy()
        image_np = np.array([0.229, 0.224, 0.225]) * image_np + np.array([0.485, 0.456, 0.406])
        image_np = np.clip(image_np, 0, 1)

        axes[row_idx, 0].imshow(image_np)
        axes[row_idx, 0].set_title('Image')
        axes[row_idx, 0].axis('off')
        
        axes[row_idx, 1].imshow(mask, cmap=cmap, vmin=0, vmax=20, interpolation='nearest')
        axes[row_idx, 1].set_title('Ground Truth')
        axes[row_idx, 1].axis('off')
        
        axes[row_idx, 2].imshow(pred, cmap=cmap, vmin=0, vmax=20, interpolation='nearest')
        axes[row_idx, 2].set_title('Prediction')
        axes[row_idx, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

# ==========================================
# 6. 主程序
# ==========================================
if __name__ == '__main__':
    # ---------------- 配置部分 ----------------
    # 请修改为你的实际路径
    root_path = r'C:\Users\Lenovo\Documents\python_code\2025ML\week7\VOCdevkit\VOC2007' 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 21 
    
    # 1. 准备数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = VOCSegmentationDataset(root_path, image_set='train', transforms=transform)
    val_dataset = VOCSegmentationDataset(root_path, image_set='val', transforms=transform)
    
    def collate_fn(batch):
        images, masks = zip(*batch)
        return torch.stack(images), torch.stack(masks)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, collate_fn=collate_fn)
    
    # 2. 初始化模型
    model = FCN_ResNet34(num_classes)
    
    # 3. 开始训练 (增加 Epoch 到 50)
    train_fcn(model, train_loader, epochs=50, lr=0.01, device=device)
    
    # 4. 可视化
    visualize_prediction(model, val_dataset, device, sample_indices=[0, 10, 20, 30])