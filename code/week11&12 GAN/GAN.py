import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# 设置超参数
batch_size = 128
lr = 0.0002
epochs = 10
latent_dim = 100
# 数据加载（以MNIST为例）
dataloader = DataLoader(
    datasets.MNIST(
        'data/',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) # 将数据归一化到[-1, 1]
        ])
    ),
    batch_size=batch_size,
    shuffle=True
)
# 定义生成器网络
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh()  # 输出范围 [-1, 1]
        )

    def forward(self, z):
        return self.model(z)
# 定义判别器网络
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()  # 输出为概率
        )

    def forward(self, x):
        return self.model(x)
# 初始化模型
G = Generator()
D = Discriminator()
# 优化器
optim_G = optim.Adam(G.parameters(), lr=lr)
optim_D = optim.Adam(D.parameters(), lr=lr)
# 损失函数
criterion = nn.BCELoss()
# 训练循环
for epoch in range(epochs):
    for idx, (real_imgs, _) in enumerate(dataloader):
        batch_size = real_imgs.size(0)
        # 真实标签与生成标签
        real_label = torch.ones((batch_size, 1))
        fake_label = torch.zeros((batch_size, 1))
        # 展平真实图像
        real_imgs = real_imgs.view(batch_size, -1)
        # ========== 训练判别器（梯度上升，实际实现为负损失的梯度下降）==========
        optim_D.zero_grad()
        # 判别真实图像
        real_output = D(real_imgs)
        loss_real = criterion(real_output, real_label)
        # 判别生成图像
        z = torch.randn(batch_size, latent_dim)
        fake_imgs = G(z).detach()  # detach防止生成器梯度更新
        fake_output = D(fake_imgs)
        loss_fake = criterion(fake_output, fake_label)
        # 判别器总损失（取负是因为最大化目标函数）
        loss_D = loss_real + loss_fake
        loss_D.backward()
        optim_D.step()
        # ========== 训练生成器（梯度下降）==========
        optim_G.zero_grad()
        # 生成新图像
        z = torch.randn(batch_size, latent_dim)
        gen_imgs = G(z)
        # 尝试欺骗判别器
        output = D(gen_imgs)
        loss_G = criterion(output, real_label)  # 希望判别器将生成的数据判断为真实
        loss_G.backward()
        optim_G.step()
        # 打印训练信息
        if idx % 100 == 0:
            print(
                f'Epoch [{epoch+1}/{epochs}], Step [{idx}/{len(dataloader)}], Loss D: '
                f'{loss_D.item():.4f}, Loss G: {loss_G.item():.4f}'
            )
