import torch
import torch.nn as nn
#————————————————————————————————————01 模型————————————————————————————————————————————————————————#
class YOLOv1(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YOLOv1, self).__init__()
        self.S = S # Grid size
        self.B = B # Number of bounding boxes
        self.C = C # Number of classes
        
        # 构建YOLOv1的卷积层和池化层
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), # Conv 64, 7x7/2
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2), # Max Pool 2x2/2
            
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1), # Conv 192, 3x3/1
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2), # Max Pool 2x2/2
            
            nn.Conv2d(192, 128, kernel_size=1, stride=1), # Conv 128, 1x1/1
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), # Conv 256, 3x3/1
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, kernel_size=1, stride=1), # Conv 256, 1x1/1
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), # Conv 512, 3x3/1
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2), # Max Pool 2x2/2
            
            nn.Conv2d(512, 256, kernel_size=1, stride=1), # Conv 256, 1x1/1
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), # Conv 512, 3x3/1
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, kernel_size=1, stride=1), # Conv 256, 1x1/1
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), # Conv 512, 3x3/1
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, kernel_size=1, stride=1), # Conv 256, 1x1/1
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), # Conv 512, 3x3/1
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 512, kernel_size=1, stride=1), # Conv 512, 1x1/1
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1), # Conv 1024, 3x3/1
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2), # Max Pool 2x2/2
            
            nn.Conv2d(1024, 512, kernel_size=1, stride=1), # Conv 512, 1x1/1
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1), # Conv 1024, 3x3/1
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 512, kernel_size=1, stride=1), # Conv 512, 1x1/1
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1), # Conv 1024, 3x3/1
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1), # Conv 1024, 3x3/1
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1), # Conv 1024, 3x3/2
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1), # Conv 1024, 3x3/1
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1), # Conv 1024, 3x3/1
            nn.LeakyReLU(0.1)
        )
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * 7 * 7, 4096), # FC1
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, self.S * self.S * (self.C + self.B * 5)) # FC2
        )
        
    def forward(self, x):
        x = self.conv_layers(x) # 通过卷积层
        x = self.fc_layers(x) # 通过全连接层
        x = x.view(-1, self.S, self.S, self.C + self.B * 5) # 调整输出形状
        return x
#_________________________________02 损失函数————————————————————————————————————————————————————#
class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20, lambda_coord=5, lambda_noobj=0.5):
        super(YoloLoss, self).__init__()
        self.S = S # 网格大小
        self.B = B # 每个网格预测的边界框数量
        self.C = C # 类别数量
        self.lambda_coord = lambda_coord # 坐标损失权重
        self.lambda_noobj = lambda_noobj # 没有物体时置信度损失权重
        
    def forward(self, predictions, target):
        # 预测的形状: (batch_size, S*S*(B*5 + C))
        predictions = predictions.view(-1, self.S, self.S, self.B * 5 + self.C)
        target = target.view(-1, self.S, self.S, self.B * 5 + self.C)
        
        # 分离预测的各个部分
        pred_boxes = predictions[..., :self.B*5].view(-1, self.S, self.S, self.B, 5)
        pred_classes = predictions[..., self.B*5:] # (batch_size, S, S, C)
        
        # 分离目标的各个部分
        target_boxes = target[..., :5].view(-1, self.S, self.S, 5)
        # 在第三个维度之后添加一个新的维度
        target_boxes = target_boxes.unsqueeze(3) # 新形状为 (batch_size, 7, 7, 1, 5)
        # 第二步: 将新添加的维度从大小1广播到大小2
        target_boxes = target_boxes.expand(-1, -1, -1, 2, -1) # 最终形状为 (batch_size, 7, 7, 2, 5)
        target_classes = target[..., self.B*5:] # (batch_size, S, S, C)
        
        # 创建掩码
        # 对象掩码: 标记哪个网格单元包含物体
        obj_mask = target[..., 4] > 0 # (batch_size, S, S)
        noobj_mask = target[..., 4] == 0 # (batch_size, S, S)
        
        # 将掩码扩展到边界框维度(batch_size, S, S, B)
        obj_mask = obj_mask.unsqueeze(-1).expand(-1, -1, -1, self.B) 
        noobj_mask = noobj_mask.unsqueeze(-1).expand(-1, -1, -1, self.B)
        
        # 计算IoU并找到负责预测的边界框
        ious = self.calculate_iou(pred_boxes[..., :4], target_boxes[..., :4]) # (batch_size, S, S, B)
        _, best_box = ious.max(dim=-1, keepdim=True) # (batch_size, S, S, 1)
        best_box = best_box.long() # 确保数据类型为long
        
        # 负责预测物体的边界框掩码
        resp_mask = torch.zeros_like(obj_mask, dtype=torch.bool) # (batch_size, S, S, B)
        resp_mask.scatter_(-1, best_box, True) # 将负责的边界框位置置为True
        
        # 坐标损失
        coord_loss = self.lambda_coord * self.coordinate_loss(pred_boxes, target_boxes, resp_mask)
        
        # 置信度损失
        conf_loss = self.confidence_loss(pred_boxes[..., 4], target_boxes[..., 4], obj_mask, noobj_mask, resp_mask)
        
        # 类别损失
        class_loss = self.class_loss(pred_classes, target_classes, obj_mask)
        
        total_loss = coord_loss + conf_loss + class_loss
        return total_loss

    def coordinate_loss(self, pred_boxes, target_boxes, resp_mask):
        # 只计算负责预测物体的边界框的坐标损失
        pred_xy = pred_boxes[..., :2][resp_mask]
        pred_wh = pred_boxes[..., 2:4][resp_mask]
        pred_wh = torch.sign(pred_wh) * torch.sqrt(torch.abs(pred_wh) + 1e-6) # 避免负值
        
        target_xy = target_boxes[..., :2][resp_mask]
        target_wh = target_boxes[..., 2:4][resp_mask]
        target_wh = torch.sqrt(target_wh)
        
        # 计算损失
        xy_loss = nn.functional.mse_loss(pred_xy, target_xy, reduction='sum')
        wh_loss = nn.functional.mse_loss(pred_wh, target_wh, reduction='sum')
        coord_loss = xy_loss + wh_loss
        return coord_loss

    def confidence_loss(self, pred_conf, target_conf, obj_mask, noobj_mask, resp_mask):
        # 负责预测物体的边界框的置信度损失
        conf_loss_obj = nn.functional.mse_loss(pred_conf[resp_mask], target_conf[resp_mask], reduction='sum')
        
        # 不负责预测物体的边界框的置信度损失，乘以lambda_noobj权重
        conf_loss_noobj = nn.functional.mse_loss(pred_conf[noobj_mask], target_conf[noobj_mask], reduction='sum')
        
        conf_loss_noobj = self.lambda_noobj * conf_loss_noobj
        
        conf_loss = conf_loss_obj + conf_loss_noobj
        return conf_loss

    def class_loss(self, pred_classes, target_classes, obj_mask):
        # 只在包含物体的网格单元计算类别损失
        obj_mask = obj_mask.any(dim=-1) # (batch_size, S, S)
        pred_classes = pred_classes[obj_mask]
        target_classes = target_classes[obj_mask]
        class_loss = nn.functional.mse_loss(pred_classes, target_classes, reduction='sum')
        return class_loss

    def calculate_iou(self, pred_boxes, target_boxes):
        # pred_boxes: (batch_size, S, S, B, 4)
        # target_boxes: (batch_size, S, S, 1, 4)
        
        # 添加目标边界框的维度以进行广播
        # 注意：这里代码逻辑与上面 forward 中 expand 过的 target_boxes 配合，通常 target_boxes 已经是 (..., 2, 4) 了
        # 如果是原始单框输入，需要 expand。这里假设输入已经对齐。
        # target_boxes = target_boxes[..., 0, :].unsqueeze(-2) 
        
        # 预测边界框的坐标
        pred_xy = pred_boxes[..., :2]
        pred_wh = pred_boxes[..., 2:4] / 2
        pred_min = pred_xy - pred_wh
        pred_max = pred_xy + pred_wh
        
        # 目标边界框的坐标
        target_xy = target_boxes[..., :2]
        target_wh = target_boxes[..., 2:4] / 2
        target_min = target_xy - target_wh
        target_max = target_xy + target_wh
        
        # 计算交集
        intersect_min = torch.max(pred_min, target_min)
        intersect_max = torch.min(pred_max, target_max)
        intersect_wh = torch.clamp(intersect_max - intersect_min, min=0)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        
        # 计算并集
        pred_area = (pred_max[..., 0] - pred_min[..., 0]) * (pred_max[..., 1] - pred_min[..., 1])
        target_area = (target_max[..., 0] - target_min[..., 0]) * (target_max[..., 1] - target_min[..., 1])
        
        union_area = pred_area + target_area - intersect_area
        
        # 计算IoU
        iou = intersect_area / (union_area + 1e-6) # 避免除以零
        
        return iou
#————————————————————————————————————03 数据集处理————————————————————————————————————————————————————————#
import torch
import xml.etree.ElementTree as ET
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image

VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
    'dog', 'horse', 'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
]

class VOCDataset(Dataset):
    def __init__(self, root, year='2007', image_set='train', S=7, B=2, C=20, transform=None, download=False):
        self.S = S # 网格大小
        self.B = B # 每个网格预测的边界框数量
        self.C = C # 类别数量
        self.transform = transform
        
        # 使用 torchvision 的 VOCDetection 加载本地数据（不自动下载）
        # 请先运行 download_voc.py 手动下载数据集
        self.voc_dataset = datasets.VOCDetection(root=root, year=year, image_set=image_set, download=download)
        
    def __len__(self):
        return len(self.voc_dataset)
    
    def __getitem__(self, idx):
        # 获取图像和对应的目标 (目标是 Pascal VOC 原始标签格式)
        image, target = self.voc_dataset[idx]
        
        # 解析目标并转换为 YOLOv1 的格式
        target = self.parse_voc_annotation(target)
        
        # 应用图像预处理
        if self.transform:
            image = self.transform(image)
            
        return image, target

    def parse_voc_annotation(self, target):
        # 获取图像的宽度和高度
        width = int(target['annotation']['size']['width'])
        height = int(target['annotation']['size']['height'])
        
        # 初始化 YOLO 格式的标签
        yolo_target = torch.zeros((self.S, self.S, self.B * 5 + self.C))
        
        for obj in target['annotation']['object']:
            class_name = obj['name']
            
            if class_name not in VOC_CLASSES:
                continue
            class_idx = VOC_CLASSES.index(class_name)
            
            # 获取物体的边界框信息 (xmin, ymin, xmax, ymax)
            bndbox = obj['bndbox']
            xmin = float(bndbox['xmin']) / width
            ymin = float(bndbox['ymin']) / height
            xmax = float(bndbox['xmax']) / width
            ymax = float(bndbox['ymax']) / height
            
            # 计算中心点坐标和宽高
            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2
            w = xmax - xmin
            h = ymax - ymin
            
            # 将中心点映射到 S x S 网格中
            grid_x = int(x_center * self.S)
            grid_y = int(y_center * self.S)
            
            # 计算中心点相对于网格单元的偏移
            x_offset = x_center * self.S - grid_x
            y_offset = y_center * self.S - grid_y
            
            # 设置标签，包括 (x, y, w, h, confidence) 和 one-hot 编码的类别
            # 置信度为1
            yolo_target[grid_y, grid_x, :5] = torch.tensor([x_offset, y_offset, w, h, 1]) 
            yolo_target[grid_y, grid_x, 10 + class_idx] = 1 # 类别的 one-hot 编码
            
        return yolo_target
#————————————————————————————————————04 训练————————————————————————————————————————————————————————#
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

# 定义图像预处理步骤
transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
])

# 创建数据集实例，使用本地数据
# 注意：请先运行 download_voc.py 下载数据集
train_dataset = VOCDataset(
    root='./data', 
    year='2007', 
    image_set='train', 
    S=7, B=2, C=20, 
    transform=transform, 
    download=False  # 禁用自动下载，使用本地数据
)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# 定义 YOLOv1 模型
model = YOLOv1() # 自定义的 YOLOv1 模型
# 定义计算设备：如果有 GPU 可用，则使用 GPU；否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 定义损失函数和优化器
criterion = YoloLoss(S=7, B=2, C=20)
optimizer = optim.SGD(model.parameters(), lr=0.00000001, momentum=0.9, weight_decay=0.0005)

# 开始训练
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for i, (images, targets) in enumerate(train_loader):
        images = images.to(device) # 输入图像
        targets = targets.to(device) # 真实标签
        
        optimizer.zero_grad()
        outputs = model(images) # 前向传播
        loss = criterion(outputs, targets) # 计算损失
        loss.backward() # 反向传播
        optimizer.step() # 更新模型参数
        
        total_loss += loss.item()
        
        if i % 10 == 0:
             print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")
             
    print(f"Epoch [{epoch+1}/{num_epochs}], Total Loss: {total_loss / len(train_loader):.4f}")

# 训练结束