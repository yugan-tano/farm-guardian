# data_collector.py
import os
import urllib.request
import zipfile


# 下载公开的植物病害数据集（示例）
def download_sample_data():
    # 创建一个示例数据集目录
    os.makedirs('data/kiwi', exist_ok=True)

    # 这里可以用PlantVillage数据集的部分图片
    # 先创建几个示例类别
    categories = ['healthy', 'ulcer', 'brown_spot', 'gray_mold']

    for category in categories:
        os.makedirs(f'data/kiwi/{category}', exist_ok=True)
        print(f"创建目录: data/kiwi/{category}")

    print("✅ 数据目录结构已创建")
    print("📁 你可以将自己的猕猴桃病害图片放到对应目录")
    print("   data/kiwi/healthy/       - 健康叶片")
    print("   data/kiwi/ulcer/         - 溃疡病")
    print("   data/kiwi/brown_spot/    - 褐斑病")
    print("   data/kiwi/gray_mold/    - 花腐病")


if __name__ == '__main__':
    download_sample_data()