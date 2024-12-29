这是一个使用PyTorch框架，采用MNIST数据集进行训练的CNN手写数字识别模型。配合PyQt5完成了手写识别小系统的界面。

# 1.数据集下载
首先需要运行 project/src/data_loader 下载数据集。包括 MNIST 数据集和 EMNIST 数据集。

数据就被放在 project/data/ 下

# 2.模型训练
运行 project/src/main.py 训练模型。

训练好的模型权重保存在 project/model/ 下

其中可以修改 data 参数，使用 MNIST 数据集训练或者使用 EMNIST 数据集训练。

# 3.模型测试
在 main.py 中的 trainer.test_model() 就是对模型的测试

# 4.运行 GUI 界面
运行 project/src/gui.py 即可运行 GUI 界面。

同理，我们在 gui.py 中修改 data 参数，使用 MNIST 数据集训练的参数或者 EMNIST 数据集训练的参数。
