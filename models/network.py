# models/network.py (修正版)
from torchvision import models
import torch.nn as nn


def get_model(config, pretrained=False):
    """
    根据配置动态获取模型。
    """
    model_name = config.MODEL_NAME
    num_classes = config.NUM_CLASSES

    print(f"[Model] Initializing {model_name} with {num_classes} classes...")

    weights = 'DEFAULT' if pretrained else None

    # --- 动态模型选择 ---
    if model_name == 'resnet50':
        model = models.resnet50(weights=weights)
        # 替换最后一层全连接层
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(weights=weights)

        # --- 关键修正 ---
        # 错误的方式: num_ftrs = model.classifier[0].in_features  (这是第一层的输入维度, 960)
        # 正确的方式: 获取原始最后一层(classifier[3])的输入维度 (这是数据流到最后时的维度, 1280)
        num_ftrs = model.classifier[3].in_features

        # 用正确的维度替换最后一层分类器
        model.classifier[3] = nn.Linear(num_ftrs, num_classes)

    else:
        raise NotImplementedError(f"Model '{model_name}' not implemented.")

    return model