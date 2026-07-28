import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import numpy as np
import os

# ==========================================
# 1. YOLOv1 模型网络结构
# ==========================================
class YOLOv1(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YOLOv1, self).__init__()
        self.S = S  # 网格大小 7x7
        self.B = B  # 每个网格的边界框数量 2
        self.C = C  # 类别数量 20 (VOC)

        # [cite_start]构建卷积层和池化层 [cite: 249]
        self.conv_layers = nn.Sequential(
            # 第一层
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            
            # 第二层
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            
            # 第三层
            nn.Conv2d(192, 128, kernel_size=1, stride=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, kernel_size=1, stride=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),

            # 第四层 (按照YOLO v1论文，4次重复的256->512卷积)
            nn.Conv2d(512, 256, kernel_size=1, stride=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, kernel_size=1, stride=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, kernel_size=1, stride=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, kernel_size=1, stride=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 512, kernel_size=1, stride=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),

            # 第五层
            nn.Conv2d(1024, 512, kernel_size=1, stride=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 512, kernel_size=1, stride=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            
            # 第六层
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
        )

        # [cite_start]全连接层 [cite: 290]
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * 7 * 7, 4096), # 文档 OCR 修正: 102477 -> 1024*7*7
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, self.S * self.S * (self.C + self.B * 5))
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        # [cite_start]调整输出形状为 (Batch, 7, 7, 30) [cite: 303]
        x = x.view(-1, self.S, self.S, self.C + self.B * 5)
        return x

# ==========================================
# 2. 代价函数 (Loss Function)
# 对应文档 Source 746 - 869
# ==========================================
class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20, lambda_coord=5, lambda_noobj=0.5):
        super(YoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def calculate_iou(self, pred_boxes, target_boxes):
        #[cite_start]# [cite: 837] 计算IOU的逻辑
        # pred_boxes: (batch, S, S, B, 4)
        # target_boxes: (batch, S, S, 1, 4) (已扩展)
        
        # 转换为左上角和右下角坐标
        pred_xy = pred_boxes[..., :2]
        pred_wh = pred_boxes[..., 2:4] / 2
        pred_min = pred_xy - pred_wh
        pred_max = pred_xy + pred_wh

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

        return intersect_area / (union_area + 1e-6)

    def forward(self, predictions, target):
        #[cite_start]# [cite: 757]
        # reshape predictions
        predictions = predictions.view(-1, self.S, self.S, self.B * 5 + self.C)
        target = target.view(-1, self.S, self.S, self.B * 5 + self.C) # 注意：这里假设target已经是构建好的YOLO格式

        # 分离预测部分
        pred_boxes = predictions[..., :self.B*5].view(-1, self.S, self.S, self.B, 5)
        pred_classes = predictions[..., self.B*5:]

        # 分离目标部分
        # target格式: (batch, S, S, B*5 + C) = (batch, S, S, 30)
        # 前B*5维是bbox数据，后C维是类别数据
        target_boxes = target[..., :self.B*5].view(-1, self.S, self.S, self.B, 5) 
        target_classes = target[..., self.B*5:]  # 类别数据从B*5开始
        # 扩展 target_boxes 以匹配 pred_boxes 的 B 维度
        target_boxes_expanded = target_boxes.expand(-1, -1, -1, self.B, -1)

        # 掩码生成
        obj_mask = target[..., 4] > 0 # (batch, S, S) 有物体的格子
        noobj_mask = target[..., 4] == 0 # (batch, S, S) 无物体的格子
        
        # 扩展掩码
        obj_mask_expanded = obj_mask.unsqueeze(-1).expand(-1, -1, -1, self.B)
        noobj_mask_expanded = noobj_mask.unsqueeze(-1).expand(-1, -1, -1, self.B)

        # [cite_start]计算 IOU 并找到负责预测的 bounding box [cite: 783]
        # 这里的 target_boxes 取前4位坐标
        ious = self.calculate_iou(pred_boxes[..., :4], target_boxes[..., :4])
        best_ious, best_box_idx = ious.max(dim=-1, keepdim=True)
        
        # 创建负责预测的掩码 resp_mask
        resp_mask = torch.zeros_like(obj_mask_expanded, dtype=torch.bool)
        # 只在有物体的格子里，IOU最大的那个框负责
        # 注意：需要配合 obj_mask 使用
        for b in range(target.size(0)):
            for i in range(self.S):
                for j in range(self.S):
                    if obj_mask[b, i, j]:
                        best_idx = best_box_idx[b, i, j]
                        resp_mask[b, i, j, best_idx] = True

        # [cite_start]=== 1. 坐标损失 [cite: 792] ===
        pred_xy = pred_boxes[..., :2][resp_mask]
        pred_wh = pred_boxes[..., 2:4][resp_mask]
        target_xy = target_boxes_expanded[..., :2][resp_mask]
        target_wh = target_boxes_expanded[..., 2:4][resp_mask]

        # [cite_start]宽高取根号 [cite: 810]
        pred_wh = torch.sqrt(torch.abs(pred_wh) + 1e-6)
        target_wh = torch.sqrt(target_wh)

        xy_loss = nn.functional.mse_loss(pred_xy, target_xy, reduction='sum')
        wh_loss = nn.functional.mse_loss(pred_wh, target_wh, reduction='sum')
        coord_loss = self.lambda_coord * (xy_loss + wh_loss)

        # [cite_start]=== 2. 置信度损失 [cite: 797] ===
        # 包含物体的损失 (target confidence = 1)
        pred_conf_obj = pred_boxes[..., 4][resp_mask]
        # 根据YOLO v1论文，有物体时置信度目标应该是1.0
        target_conf_obj = torch.ones_like(pred_conf_obj)
        conf_loss_obj = nn.functional.mse_loss(pred_conf_obj, target_conf_obj, reduction='sum')

        # 不包含物体的损失 (target confidence = 0)
        # 注意：这里包括了背景格子，以及有物体格子中非负责的那个框
        # 文档逻辑简化：使用 noobj_mask 和 resp_mask 的反面
        # 这里使用简单的逻辑：所有不负责的框都要计算 noobj loss
        noobj_final_mask = ~resp_mask 
        pred_conf_noobj = pred_boxes[..., 4][noobj_final_mask]
        target_conf_noobj = torch.zeros_like(pred_conf_noobj) # 目标是0
        
        conf_loss_noobj = self.lambda_noobj * nn.functional.mse_loss(pred_conf_noobj, target_conf_noobj, reduction='sum')
        conf_loss = conf_loss_obj + conf_loss_noobj

        # [cite_start]=== 3. 类别损失 [cite: 800] ===
        # 只需要计算有物体的格子的类别损失
        # pred_classes: (batch, S, S, C)
        obj_mask_cls = obj_mask # (batch, S, S)
        pred_cls = pred_classes[obj_mask_cls]
        target_cls = target_classes[obj_mask_cls]
        
        class_loss = nn.functional.mse_loss(pred_cls, target_cls, reduction='sum')

        total_loss = coord_loss + conf_loss + class_loss
        return total_loss

# ==========================================
# 3. 数据集处理 (VOC Dataset)
# 对应文档 Source 1015 - 1090
# ==========================================
class VOCDataset(Dataset):
    def __init__(self, root, year='2007', image_set='train', S=7, B=2, C=20, transform=None, download=False):
        self.S = S
        self.B = B
        self.C = C
        self.transform = transform
        self.VOC_CLASSES = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
            'dog', 'horse', 'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor'
        ]
        try:
            self.voc_dataset = datasets.VOCDetection(root=root, year=year, image_set=image_set, download=download)
        except:
            print("未检测到数据集，请确保已下载或设置 download=True")
            self.voc_dataset = [] # 防止报错

    def __len__(self):
        return len(self.voc_dataset)

    def __getitem__(self, idx):
        image, target = self.voc_dataset[idx]
        yolo_target = self.parse_voc_annotation(target)
        if self.transform:
            image = self.transform(image)
        return image, yolo_target

    def parse_voc_annotation(self, target):
        width = int(target['annotation']['size']['width'])
        height = int(target['annotation']['size']['height'])
        
        # 初始化 label: (S, S, B*5 + C)
        # 为了配合 Loss 函数的 split，我们构建完整的 (S, S, B*5 + C) 格式
        # 对于VOC数据集，每个网格只标注一个物体，但需要为B个预测框都提供target占位
        yolo_target = torch.zeros((self.S, self.S, self.B * 5 + self.C)) 

        objects = target['annotation']['object']
        if not isinstance(objects, list):
            objects = [objects]

        for obj in objects:
            class_name = obj['name']
            if class_name not in self.VOC_CLASSES:
                continue
            class_idx = self.VOC_CLASSES.index(class_name)
            
            bndbox = obj['bndbox']
            xmin = float(bndbox['xmin']) / width
            ymin = float(bndbox['ymin']) / height
            xmax = float(bndbox['xmax']) / width
            ymax = float(bndbox['ymax']) / height

            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2
            w = xmax - xmin
            h = ymax - ymin

            # 添加边界检查，防止越界
            grid_x = min(int(x_center * self.S), self.S - 1)
            grid_y = min(int(y_center * self.S), self.S - 1)
            
            # 相对偏移量
            x_offset = x_center * self.S - grid_x
            y_offset = y_center * self.S - grid_y

            #[cite_start]# [cite: 976] 填充 Target
            # 只有当该网格没有被赋值过时才赋值（简化逻辑）
            if yolo_target[grid_y, grid_x, 4] == 0:
                # 为第一个bounding box填充数据 (索引0-4)
                yolo_target[grid_y, grid_x, 0] = x_offset
                yolo_target[grid_y, grid_x, 1] = y_offset
                yolo_target[grid_y, grid_x, 2] = w
                yolo_target[grid_y, grid_x, 3] = h
                yolo_target[grid_y, grid_x, 4] = 1
                # 为第二个bounding box填充相同数据 (索引5-9，如果B=2)
                if self.B > 1:
                    yolo_target[grid_y, grid_x, 5] = x_offset
                    yolo_target[grid_y, grid_x, 6] = y_offset
                    yolo_target[grid_y, grid_x, 7] = w
                    yolo_target[grid_y, grid_x, 8] = h
                    yolo_target[grid_y, grid_x, 9] = 1
                # B*5之后是类别 (索引10-29，当B=2, C=20时)
                yolo_target[grid_y, grid_x, self.B * 5 + class_idx] = 1
        
        return yolo_target

# ==========================================
# 4. 训练主程序
# 对应文档 Source 1396 - 1432
# ==========================================
if __name__ == "__main__":
    # 配置
    BATCH_SIZE = 2 # 演示用，设小一点
    LEARNING_RATE = 1e-3 # 根据YOLO v1论文，使用较大的学习率
    EPOCHS = 2
    S = 7
    B = 2
    C = 20
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 模型初始化
    model = YOLOv1(S=S, B=B, C=C).to(device)
    criterion = YoloLoss(S=S, B=B, C=C)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005) # [cite: 1410]

    # --- 数据加载 ---
    use_dummy_data = True  # <--- 修改这里为 False 以使用真实下载的 VOC 数据
    
    if use_dummy_data:
        print("Using Dummy Data for testing code logic...")
        # 生成随机数据模拟图片 (Batch, 3, 448, 448) 和 标签 (Batch, 7, 7, 30)
        # 标签最后维度 30 = B*5 (2个bbox，每个5个值) + 20 (classes)
        train_loader = [
            (torch.randn(BATCH_SIZE, 3, 448, 448), torch.zeros(BATCH_SIZE, 7, 7, 30))
            for _ in range(10)
        ]
        # 手动给dummy label造一点物体，防止 Loss 全是 0
        for _, target in train_loader:
            # 第一个bbox (索引0-4)
            target[:, 3, 3, 4] = 1  # 置信度
            target[:, 3, 3, 2:4] = 0.5  # 宽高
            # 第二个bbox (索引5-9，B=2时)
            target[:, 3, 3, 9] = 1  # 置信度
            target[:, 3, 3, 7:9] = 0.5  # 宽高
            # 类别 (在B*5之后，索引10)
            target[:, 3, 3, 10] = 1  # 属于第0类
    else:
        # [cite_start]真实数据加载逻辑 [cite: 1078]
        transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
        ])
        # 请确保当前目录下有 data 文件夹，或者更改 root 路径
        train_dataset = VOCDataset(root='./data', year='2007', image_set='train', 
                                   S=S, B=B, C=C, transform=transform, download=True)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # [cite_start]--- 训练循环 [cite: 1412] ---
    model.train()
    print("Start Training...")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for i, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images) # (Batch, 7, 7, 30)
            
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i % 5 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")

        print(f"Epoch [{epoch+1}/{EPOCHS}], Avg Loss: {total_loss / len(train_loader):.4f}")
    
    print("Training Finished.")