import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
from config import Config
from models.network import get_model
import json
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


# --- 核心包装器：允许对 split 后的子集应用不同的 transform ---
class MapDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.dataset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.dataset)


# --- 绘图函数同前 ---
def plot_history(history, save_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title('Loss Curve');
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'], label='Val Acc')
    plt.title('Accuracy Curve');
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()


def train():
    config = Config()
    device = torch.device(config.DEVICE)
    os.makedirs(config.SAVE_DIR, exist_ok=True)

    # 1. 定义不同的变换集 (注意这里先不应用到 ImageFolder)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. 读取原始全量数据 (data_raw/kiwi/)
    # 注意：ImageFolder 此时不设 transform
    full_dataset = datasets.ImageFolder(root=os.path.join("data_raw", "kiwi"))

    # 获取类别信息
    classes = full_dataset.classes
    config.NUM_CLASSES = len(classes)

    # 3. 动态计算划分数量 (70% 训练, 20% 验证, 10% 测试)
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size

    # 执行随机划分
    train_subset, val_subset, test_subset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # 固定种子保证实验可重复
    )

    # 4. 为不同子集挂载对应的 transform (训练集要增强，测试集要干净)
    train_data = MapDataset(train_subset, transform=train_transform)
    val_data = MapDataset(val_subset, transform=val_test_transform)
    test_data = MapDataset(test_subset, transform=val_test_transform)

    # 5. 构建 DataLoaders
    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"📊 数据划分完成: 总计 {total_size} 张 | 训练: {train_size} | 验证: {val_size} | 测试: {test_size}")
    print(f"📂 识别到类别: {classes}")

    # 6. 初始化模型、损失函数、优化器
    model = get_model(config, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0

    # 7. 训练循环
    for epoch in range(config.EPOCHS):
        # --- Training Phase ---
        model.train()
        t_loss, t_corr = 0.0, 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * inputs.size(0)
            t_corr += torch.sum(torch.max(outputs, 1)[1] == labels.data)

        # --- Validation Phase ---
        model.eval()
        v_loss, v_corr = 0.0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                v_loss += criterion(outputs, labels).item() * inputs.size(0)
                v_corr += torch.sum(torch.max(outputs, 1)[1] == labels.data)

        # 记录指标
        history['train_loss'].append(t_loss / train_size)
        history['train_acc'].append((t_corr.double() / train_size).item())
        history['val_loss'].append(v_loss / val_size)
        history['val_acc'].append((v_corr.double() / val_size).item())

        print(f"Epoch {epoch + 1}: Train Acc {history['train_acc'][-1]:.4f} | Val Acc {history['val_acc'][-1]:.4f}")

        if history['val_acc'][-1] > best_acc:
            best_acc = history['val_acc'][-1]
            torch.save(model.state_dict(), os.path.join(config.SAVE_DIR, 'best_model.pth'))
            # 保存映射表供推理使用
            with open(os.path.join(config.SAVE_DIR, 'class_map.json'), 'w') as f:
                json.dump({i: name for i, name in enumerate(classes)}, f)

    # 8. 训练收官：保存数据、绘图、测试集最终评估
    pd.DataFrame(history).to_csv(os.path.join(config.SAVE_DIR, 'history.csv'), index=False)
    plot_history(history, config.SAVE_DIR)

    # 加载最佳模型并在测试集上做最终决战
    model.load_state_dict(torch.load(os.path.join(config.SAVE_DIR, 'best_model.pth')))
    print("\n[Final Evaluation on Test Set]")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.to(device))
            all_preds.extend(torch.max(outputs, 1)[1].cpu().numpy())
            all_labels.extend(labels.numpy())

    # 生成分类报告和混淆矩阵
    report = classification_report(all_labels, all_preds, target_names=classes)
    print(report)
    with open(os.path.join(config.SAVE_DIR, 'test_results.txt'), 'w') as f:
        f.write(report)

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=classes, yticklabels=classes)
    plt.savefig(os.path.join(config.SAVE_DIR, 'confusion_matrix.png'))
    print(f"✅ 所有训练数据、曲线图、评估报告已保存在 {config.SAVE_DIR}")


if __name__ == '__main__':
    train()