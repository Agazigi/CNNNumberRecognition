from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

# 用来对 MNIST 数据集进行数据增强
transform_mnist = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), # 将图像转换为单通道灰度图
    transforms.RandomRotation(15), # 随机旋转图像，旋转范围是 -15 度到 +15 度
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # 对图像进行仿射变换，degrees = 0代表不旋转，translate = (0.1, 0.1)代表平移 10%
    transforms.RandomResizedCrop(28, scale=(0.8, 1.0)), # 随机裁剪图像，被裁剪为 28 * 28大小，scale = (0.8, 1.0)代表裁剪的范围是 80% 到 100%
    transforms.ToTensor(), # 将图像转化为张量
    transforms.Normalize(mean=[0.5], std=[0.5]) # 将图像进行正则化，使得输入数据在 -1 到 1 之间
])

# 用来对 EMNIST 数据集进行数据增强
transform_emnist = transforms.Compose([
    transforms.ToTensor(), # 将图像转化为张量
    transforms.Normalize((0.5,), (0.5,)) # 将图像进行正则化，使得输入数据在 -1 到 1 之间
])

def load_emnist_data(batch_size=64):
    """
    :参数:batch_size 小批量大小
    :返回: train_loader, test_loader
    """
    data_dir = os.path.join('..', 'data') # 数据集存放路径，将 .. 和 data 连接起来，得到 '../data'（对不同的系统，路径的分隔符也会不同）
    train_dataset = datasets.EMNIST(root=data_dir, split='digits', train=True, download=True, transform=transform_emnist) # 读取训练集
    test_dataset = datasets.EMNIST(root=data_dir, split='digits', train=False, download=True, transform=transform_emnist) # 读取测试集
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # 创建一个数据加载器，用于加载训练集
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) # 创建一个数据加载器，用于加载测试集
    return train_loader, test_loader

def load_mnist_data(batch_size=64):
    """
    :参数:batch_size 小批量的大小
    :返回: train_loader, test_loader
    """
    data_dir = os.path.join('..', 'data') # 数据集存放路径，将 .. 和 data 连接起来，得到 '../data'（对不同的系统，路径的分隔符也会不同）
    train_dataset = datasets.MNIST(root=data_dir, train=True, transform=transform_mnist, download=True) # 读取训练集
    test_dataset = datasets.MNIST(root=data_dir, train=False, transform=transform_mnist, download=True) # 读取测试集
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # 创建一个数据加载器，用于加载训练集
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) # 创建一个数据加载器，用于加载测试集
    return train_loader, test_loader

def show_img(data, max_images=50):
    """
    :参数: data 数据集， max_images 最多显示多少张图片
    :功能: 对数据进行一部分可视化
    """
    cnt = 0 # 进行计数
    fig = plt.figure(figsize=(10, 10)) # 创建一个画布，大小为 10 * 10
    for X, y in data: # 遍历数据集，返回一个批次
        for i in range(X.size(0)): # data 返回的是一个 4 维张量，第一个维度是 batch_size (64, 1, 28, 28)
            if cnt >= max_images: # 显示 50 张图片即可
                break
            ax = fig.add_subplot(5, 10, cnt + 1) # 在 fig 上添加一个子图，cnt + 1 是子图的位置
            ax.imshow(X[i].squeeze(), cmap='gray') # 显示图片，squeeze用于去掉维度为 1 的维度（X[i] 是 [1, 28, 28]，变成 [28, 28]），cmap='gray'是显示灰度图
            ax.set_title(f"Label: {y[i].item()}") # 设置子图的标题为这个图片的真实标签
            ax.axis('off')  # 不显示坐标轴
            cnt += 1 # 计数加一
        if cnt >= max_images: # 显示 50 张图片即可
            break
    plt.show() # 显示图片

if __name__ == "__main__":
    """
    对数据集进行下载，并且可视化一部分数据
    """
    train_data_mnist, test_data_mnist = load_mnist_data() # 加载 MNIST 数据集
    train_data_emnist, test_data_emnist = load_emnist_data() # 加载 EMNIST 数据集
    show_img(train_data_mnist) # 显示 MNIST 数据集
    show_img(train_data_emnist) # 显示 EMNIST 数据集
