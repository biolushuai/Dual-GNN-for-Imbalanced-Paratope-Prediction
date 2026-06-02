import torch
print("PyTorch 版本:", torch.__version__)
print("torch 安装路径:", torch.__file__)

# 直接测试
try:
    from torch.library import custom_op
    print("✅ custom_op 存在！")
except Exception as e:
    print("❌ 错误:", e)