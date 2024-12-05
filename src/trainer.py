import torch as th
import torch.optim as optim
from tqdm import tqdm
from torch.nn import functional as F
from sklearn.metrics import confusion_matrix


# 训练器
class Trainer:
    def __init__(self, model, analyzer, train_data, test_data, device, epochs=10, lr=0.01):
        self.model = model.to(device) # 将模型移动到计算设备上
        self.train_data = train_data # 设置训练数据
        self.test_data = test_data # 设置测试数据
        self.device = device # 设置计算设备
        self.epochs = epochs # 设置训练轮数
        self.lr = lr # 设置学习率
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr) # 设置优化器，采用 SGD 优化器
        self.analyzer = analyzer # 设置分析器
    def train_epochs(self):
        """
        :功能:对模型进行训练并进行曲线分析
        """
        self.model.train() # 设置为训练模式
        epoch_list = [] # 存储轮数
        loss_list = [] # 存储损失
        acc_list = [] # 存储准确率
        for epoch in range(self.epochs): # 每一个训练轮数
            total = 0 # 统计样本总数
            correct = 0 # 统计预测正确的样本总数
            epoch_loss = 0 # 统计损失
            with tqdm(self.train_data, desc=f"第 {epoch + 1} 轮： {epoch + 1}/{self.epochs}", ncols=100) as pbar:
                                    # 使用 tqdm 包裹 train_data, 添加进度条
                for i, (X, y) in enumerate(pbar): # 遍历训练数据， i 为索引， X 为输入， y 为标签
                    X, y = X.to(self.device), y.to(self.device) # 将数据移动到计算设备上
                    self.optimizer.zero_grad() # 清空梯度
                    y_hat = self.model.forward(X) # 前向传播
                    loss = F.cross_entropy(y_hat, y) # 计算损失，使用交叉熵损失函数
                    loss.backward() # 反向传播
                    self.optimizer.step() # 更新参数

                    total += y.size(0) # 统计样本总数
                    correct += (y_hat.argmax(dim=1) == y).sum().item() # 统计预测正确的样本总数
                    epoch_loss += loss.item() # 将损失进行累加
                    pbar.set_postfix({"loss": epoch_loss / (i + 1), "acc": correct / total}) # 更新进度条
                epoch_list.append(epoch + 1) # 将轮数添加到列表中
                loss_list.append(epoch_loss / (i + 1)) # 将平均损失添加到列表中
                acc_list.append(correct / total) # 将准确率添加到列表中
        self.analyzer.plot_with_epochs(size=(10, 10),epoch_list=epoch_list, data=loss_list,
                                       title="Training Loss", ylabel="loss", xlabel="epoch",
                                       color="red", label="loss") # 绘制损失曲线
        self.analyzer.plot_with_epochs(size=(10, 10),epoch_list=epoch_list, data=acc_list,
                                       title="Training Accuracy", ylabel="acc", xlabel="epoch",
                                       color="blue", label="acc") # 绘制训练集准确率曲线

    def test_model(self):
        """
        :功能: 对模型在测试集上进行测试
        """
        self.model.eval()  # 设置为评估模式
        total = 0  # 统计样本总数
        correct = 0  # 统计预测正确的样本总数
        total_kl_div = 0  # 统计 KL 散度
        all_preds = []  # 存储所有预测的标签
        all_labels = []  # 存储所有真实的标签

        with tqdm(self.test_data, desc="测试模型", ncols=100) as pbar:
            for X, y in pbar:  # 遍历测试数据， X 为输入， y 为标签
                X, y = X.to(self.device), y.to(self.device)  # 将数据移动到计算设备上
                y_hat = self.model.forward(X)  # 前向传播
                probs = F.softmax(y_hat, dim=1)  # 计算概率
                target_dist = th.zeros_like(probs)  # 创建一个和 y_hat 相同大小的零张量
                target_dist.scatter_(1, y.view(-1, 1), 1.0)  # 将标签转换为 one-hot 编码
                kl_div = F.kl_div(F.log_softmax(y_hat, dim=1), target_dist, reduction='batchmean')  # 计算 KL 散度

                total += y.size(0)  # 统计样本总数
                correct += (y_hat.argmax(dim=1) == y).sum().item()  # 统计预测正确的样本总数
                total_kl_div += kl_div.item()  # 将 KL 散度进行累加
                # 存储真实标签和预测标签
                all_preds.extend(y_hat.argmax(dim=1).cpu().numpy()) # 将预测标签转换为 CPU 上的 NumPy 数组
                all_labels.extend(y.cpu().numpy()) # 将真实标签转换为 CPU 上的 NumPy 数组
                # 更新进度条
                pbar.set_postfix({"acc": correct / total, "kl_div": total_kl_div / (pbar.n + 1)})
        accuracy = correct / total  # 测试集准确率
        avg_kl_div = total_kl_div / len(self.test_data)  # 测试集平均 KL 散度
        print(f"测试集准确率：{accuracy}")  # 测试集准确率
        print(f"测试集平均KL散度：{avg_kl_div}")  # 测试集平均 KL 散度
        cm = confusion_matrix(all_labels, all_preds) # 计算混淆矩阵
        self.analyzer.plot_confusion_matrix(cm) # 绘制混淆矩阵
