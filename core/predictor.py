import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from models.network import get_model
from config import Config
import os


class Predictor:
    def __init__(self):
        """
        初始化预测器：
        这里为了“高效”，我们采用懒加载（Lazy Loading）或者缓存策略。
        目前的策略是：启动时先把模型结构准备好，但不加载权重，直到第一次请求（或者这里直接加载）。
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Core] Running inference on: {self.device}")

        # 1. 定义标准的图像预处理流程
        # 注意：这是提升准确率的第一道门槛，必须和训练时保持一致
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 强制缩放
            transforms.ToTensor(),  # 转为Tensor
            transforms.Normalize(  # 标准化 (ImageNet标准均值方差)
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # 2. 缓存已加载的模型，避免每次通过 crop_type 重新加载文件
        self.loaded_models = {}

    def _load_model_for_crop(self, crop_type):
        """内部方法：根据作物类型加载对应的模型权重"""
        if crop_type in self.loaded_models:
            return self.loaded_models[crop_type]

        # 获取该作物的类别标签列表
        if crop_type not in Config.CLASSES:
            raise ValueError(f"Unknown crop type: {crop_type}")

        class_names = Config.CLASSES[crop_type]
        num_classes = len(class_names)

        # 实例化网络结构
        model = get_model(num_classes=num_classes, model_name='mobilenet_v3_large', pretrained=False)

        # 加载权重文件
        model_path = Config.MODEL_PATHS.get(crop_type)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at: {model_path}")

        print(f"[Core] Loading weights from {model_path}...")

        # 这里的 map_location 保证即使只有 CPU 也能加载 GPU 训练的模型
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)

        model.to(self.device)
        model.eval()  # 开启评估模式 (关闭 Dropout 等)

        # 存入缓存
        self.loaded_models[crop_type] = model
        return model

    def predict(self, image_path, crop_type='kiwi'):
        """
        核心推理函数
        """
        try:
            # 1. 加载模型
            model = self._load_model_for_crop(crop_type)

            # 2. 处理图片
            image = Image.open(image_path).convert('RGB')  # 确保是RGB，防止PNG透明通道报错
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)  # 增加 Batch 维度 [1, 3, 224, 224]

            # 3. 推理 (No Grad 加速)
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)  # 转化为概率 [0.1, 0.8, 0.1]

                # 获取最大概率和对应的索引
                confidence, predicted_idx = torch.max(probabilities, 1)

                confidence = confidence.item()
                idx = predicted_idx.item()

                # 获取类别名称
                class_names = Config.CLASSES[crop_type]
                predicted_class = class_names[idx]

            return {
                'class_name': predicted_class,
                'confidence': float(f"{confidence:.4f}"),
                'crop': crop_type
            }

        except Exception as e:
            print(f"[Core Error] {str(e)}")
            # 这里为了系统稳定，如果出错返回一个 None 或者特定错误结构
            # 但既然 app.py 有 catch，这里直接抛出更直观
            raise e