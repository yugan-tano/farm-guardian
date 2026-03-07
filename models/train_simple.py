# models/train_with_real_data.py
import tensorflow as tf
import numpy as np
import os
from tensorflow import keras
from tensorflow.keras import layers, preprocessing
import matplotlib.pyplot as plt
from PIL import Image
import random


def create_improved_cnn(num_classes=4):
    """创建改进的CNN模型"""
    model = keras.Sequential([
        # 输入层
        layers.Input(shape=(150, 150, 3)),  # 稍微大一点

        # 数据增强
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2),

        # 归一化
        layers.Rescaling(1. / 255),

        # 第一卷积块
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        # 第二卷积块
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        # 第三卷积块
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        # 第四卷积块
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),

        # 全连接层
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        # 输出层
        layers.Dense(num_classes, activation='softmax')
    ])

    # 编译模型
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def load_real_data(data_dir='../data'):
    """加载真实图片数据"""
    print(f"📂 从 {data_dir} 加载数据...")

    categories = []
    images = []
    labels = []

    # 检查目录结构
    if not os.path.exists(data_dir):
        print(f"❌ 数据目录不存在: {data_dir}")
        return None, None

    # 获取所有子目录（病害类别）
    try:
        class_names = sorted([d for d in os.listdir(data_dir)
                              if os.path.isdir(os.path.join(data_dir, d))])
    except:
        print(f"❌ 无法读取 {data_dir} 目录")
        return None, None

    if len(class_names) == 0:
        print("❌ 数据目录中没有子目录")
        print("📁 请确保目录结构为:")
        print("   data/健康/")
        print("   data/溃疡病/")
        print("   data/褐斑病/")
        print("   data/花腐病/")
        return None, None

    print(f"✅ 找到 {len(class_names)} 个病害类别: {class_names}")

    # 遍历每个类别
    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(data_dir, class_name)

        if not os.path.isdir(class_path):
            continue

        # 获取所有图片文件
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG']:
            image_files.extend([f for f in os.listdir(class_path) if f.endswith(ext)])

        print(f"   {class_name}: {len(image_files)} 张图片")

        # 加载图片
        for img_file in image_files[:300]:  # 最多300张每类
            try:
                img_path = os.path.join(class_path, img_file)
                img = Image.open(img_path)

                # 转换为RGB
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # 调整大小
                img = img.resize((150, 150))

                # 转换为numpy数组
                img_array = np.array(img, dtype=np.float32) / 255.0

                images.append(img_array)
                labels.append(class_idx)

            except Exception as e:
                print(f"     跳过 {img_file}: {e}")

    if len(images) == 0:
        print("❌ 没有加载到任何图片")
        return None, None

    # 转换为numpy数组
    X = np.array(images)
    y = tf.keras.utils.to_categorical(np.array(labels), num_classes=len(class_names))

    print(f"✅ 成功加载 {len(images)} 张图片")
    print(f"   形状: {X.shape}")
    print(f"   类别数: {len(class_names)}")

    return X, y, class_names


def train_model():
    """训练模型"""
    print("🚀 开始用真实数据训练模型...")

    # 1. 加载真实数据
    result = load_real_data('../data/kiwi')
    if result is None:
        print("❌ 数据加载失败，使用合成数据...")
        return train_synthetic()

    X, y, class_names = result

    # 2. 打乱数据
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    # 3. 划分训练集和验证集
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    print(f"📊 数据划分:")
    print(f"   训练集: {len(X_train)} 张图片")
    print(f"   验证集: {len(X_val)} 张图片")

    # 4. 创建模型
    model = create_improved_cnn(num_classes=len(class_names))
    model.summary()

    # 5. 训练模型（更少epochs，防止过拟合）
    print("🔧 训练中...")

    # 添加早停
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    # 保存最佳模型
    checkpoint = keras.callbacks.ModelCheckpoint(
        'models/best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=16,
        epochs=30,
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )

    # 6. 保存最终模型
    model.save('models/plant_disease_model.keras')
    print("✅ 模型保存完成: models/plant_disease_model.keras")

    # 7. 保存类别名称
    with open('models/class_names.txt', 'w', encoding='utf-8') as f:
        for name in class_names:
            f.write(name + '\n')
    print("✅ 类别名称保存: models/class_names.txt")

    # 8. 绘制训练历史
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='训练准确率')
    plt.plot(history.history['val_accuracy'], label='验证准确率')
    plt.title('模型准确率')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title('模型损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('models/training_history_real.png', dpi=150)
    print("📈 训练历史图已保存: models/training_history_real.png")

    # 9. 评估模型
    print("\n📊 模型评估:")
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"   验证准确率: {val_accuracy * 100:.2f}%")
    print(f"   验证损失: {val_loss:.4f}")

    return model, class_names


def train_synthetic():
    """备用：用合成数据训练"""
    print("使用合成数据训练...")

    # 生成合成数据
    num_samples = 200
    img_size = 150

    X = np.random.rand(num_samples, img_size, img_size, 3).astype(np.float32)
    y = np.random.randint(0, 4, size=(num_samples,))
    y = tf.keras.utils.to_categorical(y, num_classes=4)

    # 划分数据集
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # 创建简单模型
    model = create_improved_cnn(num_classes=4)

    # 训练
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=16,
        epochs=10,
        verbose=1
    )

    # 保存模型
    model.save('models/plant_disease_model.keras')
    print("✅ 模型保存完成（合成数据）")

    return model, ['健康', '猕猴桃溃疡病', '猕猴桃褐斑病', '猕猴桃花腐病']


if __name__ == '__main__':
    # 设置随机种子
    print(f"{os.getcwd()}")
    np.random.seed(42)
    tf.random.set_seed(42)

    # 检查TensorFlow版本
    print(f"TensorFlow版本: {tf.__version__}")

    # 创建models目录
    os.makedirs('models', exist_ok=True)

    # 训练模型
    train_model()