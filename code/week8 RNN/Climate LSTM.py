import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
#--------------------------------1.数据预处理--------------------------------#
# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 加载数据
df = pd.read_csv('./data/jena_climate_2009_2016.csv')
# 选择特征并标准化
features = ['p (mbar)', 'T (degC)', 'rh (%)', 'VPmax (mbar)',
'VPdef (mbar)', 'sh (g/kg)', 'H2OC (mmol/mol)', 'rho (g/m**3)']
data = df[features].values.astype(np.float32)
# 数据标准化
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data).astype(np.float32)
# 划分数据集
train_size = int(len(scaled_data) * 0.7)
val_size = int(len(scaled_data) * 0.2)
train_data = scaled_data[:train_size]
val_data = scaled_data[train_size:train_size+val_size]
test_data = scaled_data[train_size+val_size:]
#--------------------------------2.创建数据集类--------------------------------#
class ClimateDataset(Dataset):
    def __init__(self, data, look_back=144, forecast_horizon=6):
        self.X, self.y = self.create_sequences(data, look_back, forecast_horizon)
    def create_sequences(self, data, look_back, forecast_horizon):
        X, y = [], []
        for i in range(len(data)-look_back-forecast_horizon):
            X.append(data[i:i+look_back])
            y.append(data[i+look_back:i+look_back+forecast_horizon, 1]) # 温度在第二列
        return torch.from_numpy(np.array(X)), torch.from_numpy(np.array(y))
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
# 参数设置
look_back = 144 # 24小时数据（144个10分钟间隔）
forecast_horizon = 6 # 预测未来1小时（6个时间步）
# 创建数据集
train_dataset = ClimateDataset(train_data, look_back, forecast_horizon)
val_dataset = ClimateDataset(val_data, look_back, forecast_horizon)
test_dataset = ClimateDataset(test_data, look_back, forecast_horizon)
# 创建DataLoader
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
#--------------------------------3.构建LSTM模型--------------------------------#
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
        batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        # LSTM层
        out, (h_n, c_n) = self.lstm(x)
        # 只取最后一个时间步的输出
        out = out[:, -1, :]
        # 全连接层
        out = self.dropout(out)
        out = self.linear(out)
        return out
# 模型参数
input_size = len(features) # 8个特征
hidden_size = 64
output_size = forecast_horizon # 预测6个时间步的温度
model = LSTMModel(input_size, hidden_size, output_size).to(device)
print(model)
#--------------------------------4.模型训练--------------------------------#
# 训练参数
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 50
best_val_loss = float('inf')
patience = 10
trigger_times = 0
# 训练记录
train_losses = []
val_losses = []
for epoch in range(epochs):
    # 训练阶段
    model.train()
    train_loss = 0
    # 使用 tqdm 包装训练数据加载器，显示进度条
    loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
    for X_batch, y_batch in loop:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    # 验证阶段
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_val, y_val in val_loader:
            X_val, y_val = X_val.to(device), y_val.to(device)
            outputs = model(X_val)
            val_loss += criterion(outputs, y_val).item()
    # 记录损失
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    
    # 打印当前 Epoch 的平均损失
    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    # 早停机制 (Early Stopping)
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
# 绘制损失曲线
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.legend()
plt.show()