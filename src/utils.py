import torch as th
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

def try_gpu(i=0):
    """
    :功能:用来获取 GPU，没有就返回 CPU，可以指定 GPU的编号
    """
    if th.cuda.device_count() >= i + 1: # 判断是否有 GPU
        return th.device(f'cuda:{i}') # 如果有的话返回指定的 GPU
    return th.device('cpu') # 如果没有 GPU 就返回 CPU


def img_to_tensor(img_path, output_size=(28, 28)):
    """
    :参数: img_path 图片路径， output_size 输出图片的大小
    :功能: 将图片转化成 MNIST 数据集的格式，并返回图像的张量
    """
    img = Image.open(img_path) # 使用 PIL 读取图片
    gray_img = img.convert('L') # 将图片转化为灰度图
    inverted_img = Image.eval(gray_img, lambda x: 255 - x) # 将图片翻转颜色，将黑色变成白色，白色变成黑色
    resized_img = inverted_img.resize(output_size, Image.Resampling.LANCZOS) # 将图片缩放为 28 * 28，其中 LANCZOS 是一种插值算法，用于缩放图片
    # 使用 matplotlib 显示处理后的图片
    # 将处理完后的图片显示出来
    plt.imshow(resized_img, cmap='gray') # 显示图片，squeeze用于去掉维度为 1 的维度（X[i] 是 [1, 28, 28]，变成 [28, 28]），cmap='gray'是显示灰度图
    plt.axis('off') # 不显示坐标轴
    plt.show() # 显示图片
    # 对图片进行预处理，包括转换为张量，归一化
    transform = transforms.Compose([
        transforms.ToTensor(), # 转换为张量
        transforms.Normalize((0.5,), (0.5,)) # 正则化，使得输入数据在 -1 到 1 之间
    ])
    tensor_img = transform(resized_img) # 将缩放后的图片转换为张量
    tensor_img = tensor_img.unsqueeze(0) # 添加一个维度，即 (1, 1, 28, 28)，用来符合模型输入的格式
    return tensor_img # 返回张量