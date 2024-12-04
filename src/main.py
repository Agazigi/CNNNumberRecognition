import utils
from model import CNNModel
from trainer import Trainer
import data_loader
import matplotlib.pyplot as plt
from analyzer import Analyzer

def main():
    device = utils.try_gpu() # 获取可用的 GPU
    print(f"正在使用处理器 {device} 进行计算")
    model = CNNModel() # 创建模型对象
    analyzer = Analyzer() # 创建分析器对象
    train_data, test_data = data_loader.load_emnist_data() # 加载数据
    trainer = Trainer(model=model, analyzer=analyzer, train_data=train_data, test_data=test_data, device=device, epochs=10, lr=0.01) # 创建训练器对象
    trainer.train_epochs() # 训练模型
    trainer.test_model() # 测试模型
    model.save_model() # 保存模型参数
    plt.show() # 显示图像

if __name__ == '__main__':
    main()