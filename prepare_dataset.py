# prepare_dataset.py
import os
import shutil
import random
from tqdm import tqdm
import argparse


def split_dataset(source_dir, dest_dir, split_ratio=0.8):
    """
    将原始数据集划分为训练集和验证集。

    :param source_dir: 原始数据集的根目录 (e.g., 'data_raw/kiwi')
    :param dest_dir: 划分后数据集的存放目录 (e.g., 'data/kiwi')
    :param split_ratio: 训练集所占的比例
    """
    print(f"--- 开始准备数据集 ---")
    print(f"源目录: {source_dir}")
    print(f"目标目录: {dest_dir}")
    print(f"划分比例 (训练集): {split_ratio:.2f}")

    if not os.path.exists(source_dir):
        print(f"错误: 源目录 '{source_dir}' 不存在！")
        return

    # 清理或创建目标目录
    if os.path.exists(dest_dir):
        print(f"警告: 目标目录 '{dest_dir}' 已存在，将先被清空。")
        shutil.rmtree(dest_dir)

    train_path = os.path.join(dest_dir, 'train')
    val_path = os.path.join(dest_dir, 'val')
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)

    # 获取所有类别
    class_names = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    if not class_names:
        print(f"错误: 在 '{source_dir}' 中没有找到任何类别子目录！")
        return

    print(f"找到类别: {class_names}")

    # 遍历每个类别进行划分
    for class_name in class_names:
        source_class_dir = os.path.join(source_dir, class_name)

        # 创建目标的类别目录
        train_class_dir = os.path.join(train_path, class_name)
        val_class_dir = os.path.join(val_path, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)

        # 获取所有图片文件
        images = [f for f in os.listdir(source_class_dir) if os.path.isfile(os.path.join(source_class_dir, f))]

        # 随机打乱
        random.shuffle(images)

        # 计算划分点
        split_point = int(len(images) * split_ratio)

        train_images = images[:split_point]
        val_images = images[split_point:]

        # 拷贝文件
        print(f"\n正在处理类别: '{class_name}' (共 {len(images)} 张图片)")
        for image in tqdm(train_images, desc=f'  -> 拷贝到训练集 ({len(train_images)} 张)'):
            shutil.copy2(os.path.join(source_class_dir, image), os.path.join(train_class_dir, image))

        for image in tqdm(val_images, desc=f'  -> 拷贝到验证集 ({len(val_images)} 张)'):
            shutil.copy2(os.path.join(source_class_dir, image), os.path.join(val_class_dir, image))

    print("\n--- ✅ 数据集准备完成！ ---")
    print(f"现在你可以运行 'python train.py' 来开始训练了。")


if __name__ == '__main__':
    # 使用 argparse 让脚本可以从命令行接收参数，更灵活
    parser = argparse.ArgumentParser(description="自动划分图像数据集")
    parser.add_argument('--source', type=str, default='data_raw/kiwi', help="原始数据集的根目录")
    parser.add_argument('--dest', type=str, default='data/kiwi', help="划分后数据集的存放目录")
    parser.add_argument('--ratio', type=float, default=0.8, help="训练集所占的比例 (例如 0.8 表示 80%)")

    args = parser.parse_args()

    split_dataset(args.source, args.dest, args.ratio)