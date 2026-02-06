import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name:        {torch.cuda.get_device_name(0)}")
else:
    print("⚠️  警告: 目前正在使用 CPU 运行，速度会很慢！")
    print("   如果是英伟达显卡，请检查 CUDA 版本与 Torch 版本是否匹配。")