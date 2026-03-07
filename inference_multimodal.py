import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import os

# --- 配置参数 ---
MODEL_PATH = 'runs/kiwi_mobilenet_v3_large/best_model.pth'
CLASS_MAP_PATH = 'runs/kiwi_mobilenet_v3_large/class_map.json'
TEST_IMAGE_PATH = 'test_kimage.jpg'

# --- 模拟传感器数据 (你可以修改这里来测试不同场景) ---
# 场景 A: 低温高湿 (溃疡病爆发环境) -> 比如设为 Temp=12, Hum=90
# 场景 B: 高温干燥 (相对安全) -> 比如设为 Temp=28, Hum=40
CURRENT_TEMP = 12.5  # 摄氏度
CURRENT_HUM = 88.0  # 相对湿度 %


def load_model(model_path, num_classes):
    print(f"正在加载模型: {model_path} ...")
    model = models.mobilenet_v3_large(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model


def get_environmental_risk(disease_label, temp, hum):
    """
    多模态融合逻辑核心：根据病害类型和环境数据，判断风险等级
    这是基于农业知识的“专家系统”规则
    """
    risk_level = "普通"
    suggestion = "保持观察。"

    if disease_label == 'ulcer':  # 溃疡病
        if 5 <= temp <= 20 and hum > 80:
            risk_level = "🔴 极高 (High Risk)"
            suggestion = "🚨 紧急预警！低温高湿极大加速溃疡扩散！立即隔离并喷施铜制剂！"
        elif hum > 70:
            risk_level = "🟠 中高 (Medium Risk)"
            suggestion = "环境湿润，利于病菌繁殖，建议加强通风。"

    elif disease_label == 'gray_mold':  # 灰霉病
        if 18 <= temp <= 23 and hum > 90:
            risk_level = "🔴 极高 (High Risk)"
            suggestion = "🚨 湿度极高，灰霉病爆发期！立即降湿并用药！"

    elif disease_label == 'brown_spot':  # 褐斑病
        if temp > 25 and hum > 75:
            risk_level = "🟠 中高 (Medium Risk)"
            suggestion = "高温高湿，注意叶面排水。"

    elif disease_label == 'healthy':
        if hum > 85:
            risk_level = "🟡 预防 (Pre-warning)"
            suggestion = "虽然当前健康，但湿度过大，建议预防性喷洒杀菌剂。"
        else:
            risk_level = "🟢哪怕 (Safe)"
            suggestion = "环境适宜，继续保持。"

    return risk_level, suggestion


def predict_multimodal(model, image_path, class_names, temp, hum):
    """
    融合推理函数
    """
    if not os.path.exists(image_path):
        print(f"找不到图片: {image_path}")
        return

    # 1. 视觉推理 (Visual Inference)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

    top_prob, top_class_idx = torch.max(probabilities, 1)
    predicted_label = class_names[str(top_class_idx.item())]
    confidence = top_prob.item() * 100

    # 2. 多模态融合 (Multi-modal Fusion)
    risk_level, advice = get_environmental_risk(predicted_label, temp, hum)

    # 3. 输出综合报告
    print("\n" + "=" * 50)
    print(f"📊 猕猴桃智慧监测系统 - 多模态诊断报告")
    print("=" * 50)
    print(f"📸 [视觉模态] 输入图像: {image_path}")
    print(f"🌡️ [传感模态] 当前环境: 温度 {temp}°C | 湿度 {hum}%")
    print("-" * 50)
    print(f"🧠 AI视觉诊断结果: \033[1;33m{predicted_label.upper()}\033[0m")  # 高亮显示
    print(f"🎯 视觉置信度:    {confidence:.2f}%")
    print("-" * 50)
    print(f"⚠️ 综合风险等级:  {risk_level}")
    print(f"💡 农业专家建议:  {advice}")
    print("=" * 50 + "\n")


if __name__ == '__main__':
    # 加载资源
    with open(CLASS_MAP_PATH, 'r') as f:
        class_names = json.load(f)

    model = load_model(MODEL_PATH, num_classes=len(class_names))

    # 运行多模态推理
    predict_multimodal(model, TEST_IMAGE_PATH, class_names, CURRENT_TEMP, CURRENT_HUM)