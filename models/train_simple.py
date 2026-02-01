# models/train_simple.py - 修复版
import tensorflow as tf
import numpy as np
import os
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


def create_simple_cnn(num_classes=4):
    """创建一个简单的CNN模型"""
    model = keras.Sequential([
        # 输入层
        layers.Input(shape=(128, 128, 3)),

        # 数据增强（训练时随机变换）
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),

        # 归一化
        layers.Rescaling(1. / 255),

        # 卷积层
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),

        # 全连接层
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),

        # 输出层（4个病害类别）
        layers.Dense(num_classes, activation='softmax')
    ])

    # 编译模型
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def generate_synthetic_data():
    """生成合成数据用于演示"""
    print("生成合成训练数据...")

    num_samples = 100
    img_size = 128

    # 创建随机图像数据
    X_train = np.random.rand(num_samples, img_size, img_size, 3).astype(np.float32)

    # 创建随机标签
    y_train = np.random.randint(0, 4, size=(num_samples,))
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=4)

    return X_train, y_train


def train_model():
    """训练模型"""
    print("开始训练模型...")

    # 1. 创建模型
    model = create_simple_cnn()
    model.summary()

    # 2. 生成数据
    X_train, y_train = generate_synthetic_data()

    # 3. 训练模型
    print("训练中...")
    history = model.fit(
        X_train, y_train,
        batch_size=16,
        epochs=5,  # 减少epochs，节省时间
        validation_split=0.2,
        verbose=1
    )

    # 4. 保存模型为Keras格式（新格式）
    try:
        model.save('models/plant_disease_model.keras')
        print("✅ 模型保存为 .keras 格式")
    except:
        # 备用：保存为H5格式
        model.save('models/plant_disease_model.h5')
        print("✅ 模型保存为 .h5 格式")

    # 5. 绘制训练历史
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='训练准确率')
    plt.plot(history.history['val_accuracy'], label='验证准确率')
    plt.title('模型准确率')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title('模型损失')
    plt.legend()

    plt.savefig('models/training_history.png')
    print("📈 训练历史图已保存")

    # 不显示图表（避免阻塞）
    # plt.show()

    return model


if __name__ == '__main__':
    # 检查TensorFlow版本
    print(f"TensorFlow版本: {tf.__version__}")

    # 创建models目录
    os.makedirs('models', exist_ok=True)

    # 训练模型
    train_model()