import matplotlib.pyplot as plt

class Analyzer:
    def __init__(self):
        pass
    def plot_with_epochs(self, size, epoch_list, data, label, xlabel, ylabel, title, color):
        """
        :参数: size 图片大小, epoch_list 轮次列表, label 标签, xlabel x轴标签, ylabel y轴标签, title 标题, color 画图颜色
        :功能: 画出训练过程曲线
        """
        plt.figure(figsize=size) # 设置图片大小
        plt.plot(epoch_list, data, label=label, color=color) # 画出曲线，同时设置标签和颜色
        plt.xlabel(xlabel) # 设置x轴标签
        plt.ylabel(ylabel) # 设置y轴标签
        plt.title(title) # 设置标题
        plt.legend() # 显示标签
        plt.grid() # 显示网格

