import torch
import sys

print(f"CUDA版本: {torch.version.cuda}")
print(f"Python版本: {sys.version}")
print(f"显卡型号: {torch.cuda.get_device_name(0)}")