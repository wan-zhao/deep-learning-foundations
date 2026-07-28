"""
测试VOC数据集是否能正确加载
"""
import torch
from torchvision import datasets, transforms

print("开始测试VOC数据集加载...")

# 定义图像预处理
transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
])

try:
    # 尝试加载数据集
    print("正在加载数据集...")
    voc_dataset = datasets.VOCDetection(
        root='./data', 
        year='2007', 
        image_set='train', 
        download=False
    )
    
    print(f"✓ 数据集加载成功！")
    print(f"✓ 数据集大小: {len(voc_dataset)} 张图片")
    
    # 测试读取第一张图片
    print("\n测试读取第一张图片...")
    image, target = voc_dataset[0]
    print(f"✓ 图片大小: {image.size}")
    print(f"✓ 标注信息键: {target['annotation'].keys()}")
    
    # 检查是否有物体
    objects = target['annotation']['object']
    if isinstance(objects, list):
        print(f"✓ 图片包含 {len(objects)} 个物体")
    else:
        print(f"✓ 图片包含 1 个物体")
    
    print("\n✓✓✓ 所有测试通过！数据集可以正常使用。")
    
except Exception as e:
    print(f"\n✗ 错误: {e}")
    print("\n请检查:")
    print("1. 数据集是否在 ./data/VOCdevkit/VOC2007/ 目录下")
    print("2. 目录结构是否完整")
