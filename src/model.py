import torch as th
from torch import nn as nn
import utils

# 卷积神经网络模型
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] # 分类的真实标签
        self.save_path = "..\\models\\model_mnist.pth" # 在这里同一设置参数文件的路径
        self.device = utils.try_gpu() # 获取可用的 GPU 设备
        # 卷积层部分，总共 4 层卷积层
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), # 输入 (batch_size, 1, 28, 28) 输出 (batch_size, 32, 28, 28)
            nn.BatchNorm2d(32), # 输出形状不变，对输入进行归一化
            nn.ReLU(), # ReLU 激活
            nn.MaxPool2d(kernel_size=2, stride=2), # 输入 (batch_size, 32, 28, 28) 输出 (batch_size, 32, 14, 14)

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 输入 (batch_size, 32, 14, 14) 输出 (batch_size, 64, 14, 14)
            nn.BatchNorm2d(64), # 输出形状不变，对输入进行归一化
            nn.ReLU(), # ReLU 激活
            nn.MaxPool2d(kernel_size=2, stride=2), # 输入 (batch_size, 64, 14, 14) 输出 (batch_size, 64, 7, 7)

            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 输入 (batch_size, 64, 7, 7) 输出 (batch_size, 128, 7, 7)
            nn.BatchNorm2d(128), # 输出形状不变，对输入进行归一化
            nn.ReLU(), # ReLU 激活
            nn.MaxPool2d(kernel_size=2, stride=2), # 输入 (batch_size, 128, 7, 7) 输出 (batch_size, 128, 3, 3)

            nn.Conv2d(128, 256, kernel_size=3, padding=1), # 输入 (batch_size, 128, 3, 3) 输出 (batch_size, 256, 3, 3)
            nn.BatchNorm2d(256), # 输出形状不变，对输入进行归一化
            nn.ReLU(), # ReLU 激活
            nn.MaxPool2d(kernel_size=2, stride=2) # 输入 (batch_size, 256, 3, 3) 输出 (batch_size, 256, 1, 1)
        )

        # 全连接层部分
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 1 * 1, 1024), # 输入 256 * 1 * 1 输出 1024
            nn.ReLU(), # ReLU 激活
            nn.Dropout(0.5), # Dropout层用来防止过拟合
            nn.Linear(1024, 512), # 输入 1024 输出 512
            nn.ReLU(), # ReLU 激活
            nn.Dropout(0.5), # Dropout层
            nn.Linear(512, 10) # 输入 512 输出 10
        )
    def forward(self, X):
        """
        :参数: X 图像的输入张量
        :输出: 预测结果张量
        """
        X.to(self.device)
        X = self.conv_layers(X) # 卷积部分
        X = X.flatten(1) # 将张量拉直为 1 维
        X = self.fc_layers(X) # 全连接部分
        return X
    def save_model(self):
        """
        :保存模型参数到 sava_path
        """
        th.save(self.state_dict(), self.save_path) # 保存参数
        print(f"模型参数成功保存到 {self.save_path}")
    def load_model(self):
        """
        :从 sava_path 中加载模型参数
        """
        self.load_state_dict(th.load(self.save_path, map_location=self.device, weights_only=True)) # 加载参数，并且将参数移动到 GPU，并且只加载参数，不加载模型结构
        print(f"从 {self.save_path} 中成功加载模型参数")

    def predict(self, img_tensor):
        """
        :参数: 图像的输入张量
        :功能: 返回预测结果
        """
        img_tensor = img_tensor.to(self.device) # 将张量移动到 GPU
        with th.no_grad(): # 禁用梯度计算
            output = self.forward(img_tensor) # 前向传播，得到一个 10 维向量，预测标签
            label = th.argmax(output, dim=1).item() # 获取最大值对应的索引，即为预测结果
        return label  # 返回预测结果