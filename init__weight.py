import torch
import os
from config import Config
from models.network import get_model
# --- 新增：强制指定下载路径到当前项目目录下 ---
os.environ['TORCH_HOME'] = os.path.join(os.path.dirname(__file__), 'tmp_cache')

def generate_dummy_weights():
    print("正在生成初始化权重文件，以防止系统报错...")

    for crop_type, names in Config.CLASSES.items():
        save_path = Config.MODEL_PATHS[crop_type]

        # 如果文件不存在，我们才生成
        if not os.path.exists(save_path):
            print(f"正在为 [{crop_type}] 生成临时模型...")
            # 实例化一个结构
            model = get_model(num_classes=len(names), model_name='mobilenet_v3_large', pretrained=True)

            # 保存权重
            torch.save(model.state_dict(), save_path)
            print(f"  -> 已保存至: {save_path}")
        else:
            print(f"  -> [{crop_type}] 权重文件已存在，跳过。")


if __name__ == "__main__":
    generate_dummy_weights()
    print("\n✅ 所有必要文件已就绪，现在可以运行 app.py 了！")