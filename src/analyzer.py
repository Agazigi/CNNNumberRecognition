import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
    def plot_confusion_matrix(self, cm, classes=None, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
        """
        :参数: cm 混淆矩阵， classes 类名， normalize 是否归一化， title 标题， cmap 颜色
        :功能: 画出混淆矩阵图
        """
        if classes is None: # 如果没有指定类名，则使用默认的类名
            classes = [str(i) for i in range(10)]  # 对于手写数字，类的名称为 0 到 9
        if normalize: # 如果需要归一化
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 对每一行进行归一化
        plt.figure(figsize=(8, 6)) # 设置图片大小
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap=cmap,
                    xticklabels=classes, yticklabels=classes, cbar=False, linewidths=0.5, linecolor='black')
        # cm 是混淆矩阵，annot=True 表示显示数值，fmt 如果归一化则保留两位小数，否则为整数
        # cmap=cmap 表示颜色，xtickslabel 是类名，ytickslabel一样
        # cbar=False 表示不显示颜色条，linewidths=0.5 表示线宽，linecolor='black' 表示线颜色
        plt.title(title) # 设置标题
        plt.ylabel('True Label') # 设置真实标签
        plt.xlabel('Predicted Label') # 设置预测标签
