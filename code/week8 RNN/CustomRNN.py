import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# 自定义RNN类
# 自定义RNN类
class CustomRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CustomRNN, self).__init__()
        self.hidden_size = hidden_size
        # 定义权重矩阵
        self.Wxa = nn.Parameter(torch.randn(input_size, hidden_size)) # 输入到隐藏层的权重
        self.Waa = nn.Parameter(torch.randn(hidden_size, hidden_size)) # 隐藏层到隐藏层的权重
        self.Wah = nn.Parameter(torch.randn(hidden_size, output_size)) # 隐藏层到输出层的权重
        # 定义偏置
        self.ba = nn.Parameter(torch.zeros(hidden_size)) # 隐藏层偏置
        self.by = nn.Parameter(torch.zeros(output_size)) # 输出层偏置
    def forward(self, x):
        # 初始化隐藏状态
        a_t = torch.zeros(x.size(0), self.hidden_size)
        # RNN的计算过程
        for t in range(x.size(1)): # 遍历时间步
            x_t = x[:, t, :] # 获取当前时间步的输入
            a_t = torch.tanh(torch.mm(x_t, self.Wxa) + torch.mm(a_t, self.Waa) + self.ba) #计算当前隐藏状态
        # 输出层计算
        y_t = torch.mm(a_t, self.Wah) + self.by
        return y_t
# 数据加载和预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True,transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 超参数设置
input_size = 28 # MNIST图像的宽度
hidden_size = 128
output_size = 10 # 10个数字类别
learning_rate = 0.001

# 初始化模型、损失函数和优化器
model = CustomRNN(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练过程
num_epochs = 5
for epoch in range(num_epochs):
    for images, labels in train_loader:
        # MNIST数据集的每个图像是28x28的，转成[batch_size, seq_len, input_size]
        images = images.view(-1, 28, 28) # 重新调整尺寸， 28个时间步，每步28个特征（像素值）
        # 清空梯度
        optimizer.zero_grad()
        # 前向传播
        outputs = model(images)
        # 计算损失
        loss = criterion(outputs, labels)
        # 反向传播和优化
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
print("训练完成！ ")