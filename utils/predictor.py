# utils/predictor.py - 简化版
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import random


class DiseasePredictor:
    def __init__(self):
        """初始化预测器"""
        self.class_names = ['健康', '猕猴桃溃疡病', '猕猴桃褐斑病', '猕猴桃花腐病']
        self.model = None

        # 尝试加载模型
        self.load_model()

        # 建议数据库
        self.advice_db = {
            '健康': '作物健康状况良好，继续保持当前管理措施。定期检查，预防为主。',
            '猕猴桃溃疡病': '立即隔离病株！喷洒50%多菌灵800倍液，连续3天。清除病残体，冬季清园。',
            '猕猴桃褐斑病': '剪除病叶集中销毁，喷洒70%代森锰锌600倍液，7-10天一次。改善通风透光条件。',
            '猕猴桃花腐病': '摘除病花，喷洒50%速克灵1500倍液。增施磷钾肥，提高抗病力。控制氮肥。'
        }

    def load_model(self):
        """尝试加载模型"""
        model_paths = [
            'models/plant_disease_model.keras',
            'models/plant_disease_model.h5'
        ]

        for path in model_paths:
            if os.path.exists(path):
                try:
                    self.model = tf.keras.models.load_model(path)
                    print(f"✅ AI模型加载成功: {path}")
                    return True
                except Exception as e:
                    print(f"❌ 模型加载失败 {path}: {e}")

        print("⚠️ 未找到可用模型，使用模拟模式")
        return False

    def preprocess_image(self, image_path, target_size=(128, 128)):
        """预处理图像"""
        try:
            img = Image.open(image_path)

            if img.mode != 'RGB':
                img = img.convert('RGB')

            img = img.resize(target_size)
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            return img_array
        except Exception as e:
            print(f"图像预处理失败: {e}")
            return None

    def predict(self, image_path):
        """预测病害"""
        # 如果有模型，使用模型预测
        if self.model is not None:
            try:
                img_array = self.preprocess_image(image_path)
                if img_array is not None:
                    predictions = self.model.predict(img_array, verbose=0)
                    predicted_class = np.argmax(predictions[0])
                    confidence = float(predictions[0][predicted_class])
                    disease_name = self.class_names[predicted_class]

                    return self._create_report(disease_name, confidence, '真实AI模型')
            except Exception as e:
                print(f"AI推理失败: {e}")

        # 否则使用模拟
        return self.predict_mock(image_path)

    def predict_mock(self, image_path):
        """模拟预测"""
        # 80%概率返回溃疡病（为了演示）
        if random.random() < 0.8:
            disease_name = '猕猴桃溃疡病'
        else:
            disease_name = random.choice(self.class_names)

        confidence = round(random.uniform(0.75, 0.95), 2)

        return self._create_report(disease_name, confidence, '模拟模式')

    def _create_report(self, disease_name, confidence, ai_model):
        """创建统一格式的报告"""
        return {
            'disease': disease_name,
            'confidence': confidence,
            'risk_level': self.get_risk_level(disease_name, confidence),
            'advice': self.advice_db.get(disease_name, '请咨询专业农技人员'),
            'ai_model': ai_model
        }

    def get_risk_level(self, disease, confidence):
        """根据病害和置信度确定风险等级"""
        if disease == '健康':
            return '低'
        elif confidence > 0.8:
            return '高'
        elif confidence > 0.6:
            return '中'
        else:
            return '低'


# 创建全局预测器实例
predictor = DiseasePredictor()