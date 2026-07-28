import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 设置中文字体（避免乱码）
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
mpl.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 参数设置
lr = 0.001
batch_size = 128
epochs = 16

# 加载数据
train_X, test_X, train_y, test_y = np.load('mnist.npy', allow_pickle=True)

# 转换为tensor
x_train = torch.tensor(train_X, dtype=torch.float32)
y_train = torch.tensor(train_y)
x_test = torch.tensor(test_X, dtype=torch.float32)
y_test = torch.tensor(test_y)

# 创建Dataset
train_ds = TensorDataset(x_train, y_train)
test_ds = TensorDataset(x_test, y_test)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

# LeNet网络构建
LeNet5 = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=0),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    nn.Conv2d(6, 16, kernel_size=5, padding=0),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    nn.Flatten(),
    nn.Linear(256, 120),#这里是256，而不是400！！！！
    nn.ReLU(),
    nn.Linear(120, 84),
    nn.ReLU(),
    nn.Linear(84, 10)
)

# 训练设置
import torch.nn.functional as F
loss_func = F.cross_entropy
opt = torch.optim.SGD(LeNet5.parameters(), lr=lr)

# 初始化记录列表
train_losses = []
train_accs = []
test_accs = []

# 训练循环
for epoch in range(epochs):
    # 1. 训练阶段
    LeNet5.train()
    epoch_loss = 0.0
    correct = 0
    total = 0
    
    for xb, yb in train_dl:
        xb = xb.reshape(-1, 1, 28, 28)
        
        # 前向传播
        pred = LeNet5(xb)
        loss = loss_func(pred, yb)
        
        # 反向传播
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        # 累计损失和准确率
        epoch_loss += loss.item()
        _, predicted = torch.max(pred, 1)
        total += yb.size(0)
        correct += (predicted == yb).sum().item()
    
    # 计算当前epoch的训练损失和准确率
    avg_train_loss = epoch_loss / len(train_dl)
    train_acc = correct / total
    train_losses.append(avg_train_loss)
    train_accs.append(train_acc)
    
    # 2. 测试阶段
    LeNet5.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for xb_test, yb_test in test_dl:
            xb_test = xb_test.reshape(-1, 1, 28, 28)
            pred_test = LeNet5(xb_test)
            
            _, predicted_test = torch.max(pred_test, 1)
            test_total += yb_test.size(0)
            test_correct += (predicted_test == yb_test).sum().item()
    
    test_acc = test_correct / test_total
    test_accs.append(test_acc)
    
    # 打印结果
    print(f"Epoch {epoch+1}/{epochs}, "
          f"Train Loss: {avg_train_loss:.4f}, "
          f"Train Acc: {train_acc:.4%}, "
          f"Test Acc: {test_acc:.4%}")


plt.figure(figsize=(10, 6))

# 绘制训练损失
plt.plot(train_losses, 'b-', label='训练损失')
# 绘制训练准确率
plt.plot(train_accs, 'g-', label='训练准确率')
# 绘制测试准确率
plt.plot(test_accs, 'r-', label='测试准确率')

# 添加标签和标题
plt.title('LeNet-5 训练指标 (MNIST)', fontsize=14)
plt.xlabel('训练轮次', fontsize=12)
plt.ylabel('指标值', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

# 添加数值标签
for i in range(epochs):
    plt.annotate(f'{train_accs[i]:.2%}', 
                 (i, train_accs[i]),
                 textcoords="offset points", 
                 xytext=(0,10), 
                 ha='center')
    plt.annotate(f'{test_accs[i]:.2%}', 
                 (i, test_accs[i]),
                 textcoords="offset points", 
                 xytext=(0,-15), 
                 ha='center')

plt.tight_layout()
plt.savefig('mnist_le_net5_training.png', dpi=150)
plt.show()

print("\n✅ 训练完成！图表已保存为 'mnist_le_net5_training.png'")
print(f"最终测试准确率: {test_accs[-1]:.4%}")