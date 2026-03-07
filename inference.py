import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import os

# --- 配置参数 ---
# 这里填写你刚才训练保存的模型路径
MODEL_PATH = 'runs/kiwi_mobilenet_v3_large/best_model.pth'
# 类别映射文件路径
CLASS_MAP_PATH = 'runs/kiwi_mobilenet_v3_large/class_map.json'
# 你想要测试的图片路径 (你可以随便改这一行读取不同的图片)
TEST_IMAGE_PATH = 'test_image.jpg'


def load_model(model_path, num_classes):
    """
    加载训练好的 MobileNetV3 模型结构和权重
    """
    print(f"正在加载模型: {model_path} ...")

    # 1. 重新构建模型结构 (必须与训练时完全一致)
    model = models.mobilenet_v3_large(weights=None)  # 推理时不需预训练权重，因为我们要加载自己的

    # 2. 修改最后的全连接层，使其符合我们的类别数量
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    # 3. 加载我们训练好的权重
    # map_location='cpu' 确保即使你在GPU训练，也可以在没有GPU的电脑(或树莓派)上运行
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    # 4. 设置为评估模式 (这很重要！会关闭Dropout等训练专用层)
    model.eval()

    print("模型加载成功！")
    return model


def predict_image(model, image_path, class_names):
    """
    读取图片并进行预测
    """
    if not os.path.exists(image_path):
        print(f"错误: 找不到图片文件 {image_path}")
        return

    # 1. 图片预处理 (必须与训练时的验证集预处理完全一致)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. 打开图片
    image = Image.open(image_path).convert('RGB')

    # 3. 应用预处理并增加一个batch维度 ( [C, H, W] -> [1, C, H, W] )
    input_tensor = transform(image).unsqueeze(0)

    # 4. 推理
    with torch.no_grad():  # 推理时不需要计算梯度，节省内存
        outputs = model(input_tensor)
        # 使用 Softmax 将输出转换为概率分布
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

    # 5. 获取结果
    top_prob, top_class_idx = torch.max(probabilities, 1)

    # 转换数据类型以便打印
    predicted_idx = top_class_idx.item()
    confidence = top_prob.item() * 100
    predicted_label = class_names[str(predicted_idx)]

    print("\n" + "=" * 30)
    print(f"🖼️  测试图片: {image_path}")
    print(f"🧠  AI 预测结果: [{predicted_label}]")
    print(f"📊  置信度 (概率): {confidence:.2f}%")
    print("=" * 30 + "\n")

    # (可选) 打印所有类别的概率
    print("各类别详细概率:")
    for idx, prob in enumerate(probabilities[0]):
        label = class_names[str(idx)]
        print(f"  - {label}: {prob.item() * 100:.2f}%")


if __name__ == '__main__':
    # 1. 加载类别映射
    if os.path.exists(CLASS_MAP_PATH):
        with open(CLASS_MAP_PATH, 'r', encoding='utf-8') as f:
            class_names = json.load(f)
        print(f"加载了 {len(class_names)} 个类别: {list(class_names.values())}")
    else:
        print(f"错误: 找不到类别映射文件 {CLASS_MAP_PATH}")
        exit()

    # 2. 检查模型文件
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 找不到模型文件 {MODEL_PATH}")
        exit()

    # 3. 加载模型
    # 这里的 len(class_names) 自动获取类别数量，非常灵活
    model = load_model(MODEL_PATH, num_classes=len(class_names))

    # 4. 进行预测
    predict_image(model, TEST_IMAGE_PATH, class_names)