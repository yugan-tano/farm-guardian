# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
from config import Config
from models.network import get_model
import json

def train():
    # 0. 加载配置
    config = Config()
    device = torch.device(config.DEVICE)

    # 1. 数据预处理和增强
    # 对训练集进行旋转、翻转等操作，增加数据多样性
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # 对验证集只做最基础的调整
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. 加载数据集
    train_dataset = datasets.ImageFolder(os.path.join(config.DATA_ROOT, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(config.DATA_ROOT, 'val'), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"训练集图像数: {len(train_dataset)}, 验证集图像数: {len(val_dataset)}")
    print(f"类别: {train_dataset.classes}")
    # 更新config中的类别数
    config.NUM_CLASSES = len(train_dataset.classes)

    # 3. 初始化模型、损失函数和优化器
    model = get_model(config, pretrained=True).to(device) # 使用预训练权重加速
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # 4. 开始训练循环
    best_acc = 0.0
    os.makedirs(config.SAVE_DIR, exist_ok=True)

    print("--- 开始训练 ---")
    for epoch in range(config.EPOCHS):
        # --- 训练模式 ---
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.EPOCHS} [Train]")
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            train_bar.set_postfix(loss=f'{loss.item():.4f}')

        epoch_loss = running_loss / len(train_dataset)

        # --- 验证模式 ---
        model.eval()
        corrects = 0
        val_loss = 0.0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{config.EPOCHS} [Val]")
        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                val_loss += loss.item() * inputs.size(0)
                corrects += torch.sum(preds == labels.data)

        epoch_val_loss = val_loss / len(val_dataset)
        epoch_acc = corrects.double() / len(val_dataset)

        print(f"Epoch {epoch + 1}/{config.EPOCHS} | Train Loss: {epoch_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_acc:.4f}")

        # 保存最好的模型
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), os.path.join(config.SAVE_DIR, 'best_model.pth'))
            print(f"新的最佳模型已保存。准确率: {best_acc:.4f}")
            class_map = {i: name for i, name in enumerate(train_dataset.classes)}
            map_path = os.path.join(config.SAVE_DIR, 'class_map.json')
            with open(map_path, 'w', encoding='utf-8') as f:
                json.dump(class_map, f, ensure_ascii=False, indent=4)
            print(f"类别映射文件已保存到: {map_path}")

    print("--- 训练完成 ---")
    print(f"最好的验证准确率是: {best_acc:.4f}")

if __name__ == '__main__':
    train()