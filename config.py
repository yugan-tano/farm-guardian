import os

# 项目根目录
BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    # 密钥，用于Session加密（生产环境需更复杂）
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'smart-agro-secret-key-2024'

    # 文件上传配置
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 限制最大上传16MB

    # 模型与标签配置
    # 注意：这里的标签顺序必须与训练模型时的 index 对应，目前基于你 data/sample 的目录结构
    CLASSES = {
        'kiwi': ['brown_spot', 'gray_mold', 'healthy', 'ulcer'],  # 猕猴桃四类
        'apple': ['rust', 'scab', 'healthy']  # 苹果 (示例，可后续补充)
    }

    # 模型文件路径
    MODEL_PATHS = {
        'kiwi': os.path.join(BASE_DIR, 'models', 'kiwi_mobilenet.pth'),
        'apple': os.path.join(BASE_DIR, 'models', 'apple_mobilenet.pth')
    }