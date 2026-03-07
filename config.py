# config.py
import torch


class Config:
    def __init__(self):
        # 1. 数据集相关配置
        self.CROP_TYPE = 'kiwi'  # 当前处理的农作物类型
        # 数据集根目录，train.py 会在这里寻找 train 和 val 文件夹
        self.DATA_ROOT = f'data/{self.CROP_TYPE}'

        # 2. 模型相关配置
        # 模型名称mobilenet_v3_large，之前用的resnet与 models/network.py 中的 get_model 函数对应
        self.MODEL_NAME = 'mobilenet_v3_large'
        # 类别数量 (这个值会在 train.py 中被自动更新，这里给个默认值)
        self.NUM_CLASSES = 2
        # 模型权重保存目录
        self.SAVE_DIR = f'runs/{self.CROP_TYPE}_{self.MODEL_NAME}'
        # 预训练模型的权重文件路径 (predictor.py 会使用)
        self.WEIGHTS_PATH = f'{self.SAVE_DIR}/best_model.pth'
        # 类别名称映射文件路径 (predictor.py 会使用)
        self.CLASS_MAP_PATH = f'{self.SAVE_DIR}/class_map.json'

        # 3. 训练过程相关配置
        # 自动选择设备: 如果有可用的 CUDA GPU，则使用 'cuda'，否则使用 'cpu'
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        # 训练的总轮数 (Epochs)
        self.EPOCHS = 50
        # 每批次处理的图片数量 (Batch Size)，如果显存不足可以适当调小
        self.BATCH_SIZE = 16
        # 学习率 (Learning Rate)
        self.LEARNING_RATE = 0.001


# 创建一个全局配置实例，方便其他文件直接 import 使用
# from config import config
config = Config()