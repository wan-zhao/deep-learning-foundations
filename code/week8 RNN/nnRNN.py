import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# 定义基于 nn.RNN 的模型
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        # RNN 输出包含输出序列和最终隐藏状态
        out, _ = self.rnn(x)
        # 使用最后一个时间步的输出进行分类
        out = self.fc(out[:, -1, :])
        return out
# 数据加载和预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True,transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 参数设置
input_size = 28
hidden_size = 128
output_size = 10
num_layers = 2
learning_rate = 0.001
# 初始化模型、损失函数和优化器
model = RNNModel(input_size, hidden_size, output_size, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# 训练模型
num_epochs = 5
for epoch in range(num_epochs):
    for images, labels in train_loader:
        images = images.view(-1, 28, 28)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
print("训练完成！ ")