import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QFileDialog, QInputDialog, QAction, QVBoxLayout, \
    QPushButton, QLabel
from PyQt5.QtGui import QIcon, QPixmap, QPainter, QPen, QColor, QFont
from PyQt5.QtCore import Qt
import time
from model import CNNModel
import utils
import os
import argparse


"""
模型加载
"""
parser = argparse.ArgumentParser()
parser.add_argument('--data', default='mnist', help='选择数据集')
args, unknown = parser.parse_known_args()
data = args.data
print(f"正在使用在 {data} 数据集上训练的模型")
model = CNNModel(data)  # 创建模型对象
model.load_model()  # 加载模型参数
model.to(model.device)  # 将模型移动到 GPU

# 画板
class Painting(QWidget): # 继承 QWidget， QWidget 是抽象类，所有窗口的父类，所有窗口都继承自 QWidget
    x0 = 0 # 鼠标按下时的 x 坐标
    y0 = 0 # 鼠标按下时的 y 坐标
    x1 = 0 # 鼠标松开时的 x 坐标
    y1 = 0 # 鼠标松开时的 y 坐标
    flag = False # 是否正在绘制
    mode = 'pen' # 默认为画笔模式，还可以改变成橡皮擦模式
    def __init__(self, parent):
        super(Painting, self).__init__(parent)
        self.pixmap = QPixmap(597, 497) # 创建一个大小为 597 x 497 的画布，其中 QPixmap 是 QImage 的子类，用于存储图像数据
        self.pixmap.fill(Qt.white) # 将画板的背景填充为白色
        self.setStyleSheet("border: 2px solid blue") # 设置边框的格式，为 2 像素、实线、蓝色的边框
        self.Color = Qt.black # 设置画笔颜色为黑色
        self.pen_width = 15 # 设置画笔的宽度为 10 像素
    def paintEvent(self, event): # 处理绘制事件
        painter = QPainter(self.pixmap) # 创建一个画笔对象
        if self.mode == 'pen': # 如果当前是“画笔”模式
            painter.setPen(QPen(self.Color, self.pen_width, Qt.SolidLine)) #
        elif self.mode == 'eraser': # 如果当前是“橡皮擦”模式
            painter.setPen(QPen(Qt.white, self.pen_width, Qt.SolidLine)) # 设置橡皮擦为白色
        if self.flag: # 如果正在绘制
            painter.drawLine(self.x0, self.y0, self.x1, self.y1) # 以按下和松开的坐标画线

        label_painter = QPainter(self) # 创建一个画笔对象
        label_painter.drawPixmap(2, 2, self.pixmap) # 绘制画布
    def mousePressEvent(self, event): # 处理鼠标按下事件
        self.x1 = event.x() # 获取鼠标当前位置的 x 坐标
        self.y1 = event.y() # 获取鼠标当前位置的 y 坐标
        self.flag = True # 设置为正在绘制
    def mouseMoveEvent(self, event): # 处理鼠标移动事件
        if self.flag: # 如果正在绘制
            self.x0 = self.x1 # 保存按下时的 x 坐标
            self.y0 = self.y1 # 保存按下时的 y 坐标
            self.x1 = event.x() # 保存松开的 x 坐标
            self.y1 = event.y() # 保存松开的 y 坐标
            self.update()  # 更新绘制
    def mouseReleaseEvent(self, event): # 处理鼠标松开事件
        self.flag = False # 设置为不绘制

# 主窗口
class Board(QMainWindow):
    def __init__(self):
        super(Board, self).__init__()
        self.result_label = None
        self.eraser_button = None
        self.pen_button = None
        self.button_width = None
        self.button_file = None
        self.brd = None
        self.predict_button = None
        self.init_ui()
    def init_ui(self):
        """
        :功能: 用来初始化界面
        """
        self.setWindowTitle("手写数字识别") # 设置窗口的标题
        icon_path = os.path.join("..", "img", "number.png")
        self.setWindowIcon(QIcon(icon_path))
        self.resize(800, 600) # 设置窗口的大小

        self.brd = Painting(self) # 创建一个画板
        self.brd.setGeometry(20, 40, 601, 501) # 设置画板的位置和尺寸

        self.button_file = QPushButton("选择文件",self) # 创建一个按钮
        self.button_file.setGeometry(660, 80, 100, 50) # 设置按钮的位置和尺寸
        self.button_file.setFont(QFont("宋体", 10)) # 设置按钮的字体和大小

        self.button_width = QPushButton("画笔宽度",self) # 创建一个按钮
        self.button_width.setGeometry(660, 460, 100, 50) # 设置按钮的位置和尺寸
        self.button_width.setFont(QFont("宋体", 10)) # 设置按钮的字体和大小

        # 画笔和橡皮按钮
        self.pen_button = QPushButton("画笔", self) # 创建一个按钮
        self.pen_button.setGeometry(660, 140, 100, 50) # 设置按钮的位置和尺寸
        self.pen_button.setFont(QFont("宋体", 10)) # 设置按钮的字体和大小

        # 新增预测按钮
        self.predict_button = QPushButton("预测", self) # 创建一个按钮
        self.predict_button.setGeometry(660, 260, 100, 50) # 设置按钮的位置和尺寸
        self.predict_button.setFont(QFont("宋体", 10)) # 设置按钮的字体和大小

        # 橡皮按钮
        self.eraser_button = QPushButton("橡皮", self) # 创建一个按钮
        self.eraser_button.setGeometry(660, 200, 100, 50) # 设置按钮的位置和尺寸
        self.eraser_button.setFont(QFont("宋体", 10)) # 设置按钮的字体大小

        # 新增清除按钮
        self.clear_button = QPushButton("一键清除", self)  # 创建清除按钮
        self.clear_button.setGeometry(660, 320, 100, 50)  # 设置按钮的位置和尺寸
        self.clear_button.setFont(QFont("宋体", 10))  # 设置按钮的字体和大小

        # 对每一个按钮连接点击事件
        self.pen_button.clicked.connect(self.use_pen)
        self.eraser_button.clicked.connect(self.use_eraser)
        self.button_file.clicked.connect(self.open_file)
        self.button_width.clicked.connect(self.choose_width)
        self.predict_button.clicked.connect(self.save_and_predict)
        self.clear_button.clicked.connect(self.clear_canvas)

        # 在界面下添加预测结果标签
        self.result_label = QLabel(self) # 创建一个标签
        self.result_label.setGeometry(20, 550, 760, 40) # 设置标签的位置和尺寸
        self.result_label.setFont(QFont("宋体", 12)) # 设置标签的字体大小
        self.result_label.setText("当然没有进行预测，请书写数字后点击预测。") # 设置默认的提示文本
    def open_file(self):
        """
        :功能: 按照指定的路径打开文件夹，然后可以选择图片到画布上
        """
        file_name = QFileDialog.getOpenFileName(self, "选择图片文件", os.path.join("..", "img", "prediction")) # 打开文件选择对话框
        if file_name[0]: # 如果选择了文件
            self.brd.pixmap = QPixmap(file_name[0]) # 设置画布的图片
    def use_pen(self):
        """
        :功能: 切换到画笔模式
        """
        self.brd.mode = 'pen' # 设置画笔模式
        self.brd.setCursor(Qt.CrossCursor) # 设置十字光标
        self.brd.Color = Qt.black # 设置画笔颜色为黑色
    def use_eraser(self):
        """
        :功能: 切换到橡皮擦模式
        """
        self.brd.mode = 'eraser' # 设置橡皮擦模式
        self.brd.setCursor(Qt.CrossCursor) # 设置十字光标
        self.brd.Color = Qt.white # 设置橡皮擦为白色
    def choose_width(self):
        """
        :功能: 选择画笔宽度
        """
        width, ok = QInputDialog.getInt(self, '选择画笔粗细', '请输入粗细：', min=1, step=1) # 弹出输入对话框，设置最小值和步长，得到一个宽度和确认状态
        if ok: # 如果确认
            self.brd.pen_width = width # 重新设置画笔宽度
    def save_and_predict(self):
        """
        :功能: 将画布上的图片存取下来，并且对图片进行预测
        """
        global model
        #  弹出文件保存对话框，得到文件路径和文件类型
        file_path, file_type = QFileDialog.getSaveFileName(self, "保存文件", "..\\img\\prediction\\" + str(time.time()) + ".png",
                                                   "PNG Files (*.png);;All Files (*)")
        if file_path:
            self.brd.pixmap.save(file_path) # 将图片保存下来
            print(f"图片已保存到: {file_path}")
            img_tensor = utils.img_to_tensor(file_path) # 依据图像的路径，调用写的工具方法将图片转化为张量
            prediction = model.predict(img_tensor) # 使用训练好的模型对图片进行预测
            print(f"图片预测为: {prediction}")
            self.result_label.setText(f"预测结果: {prediction}") # 显示预测结果
    def clear_canvas(self):
        """
        :功能: 清空画布
        """
        self.brd.pixmap.fill(Qt.white) # 填充画布为白色
        self.brd.update() # 更新画布，重新显示


if __name__ == '__main__':
    app = QApplication(sys.argv) # 创建一个应用程序对象
    board = Board() # 创建一个画板对象
    board.show() # 显示画板对象
    sys.exit(app.exec_()) # 运行程序
