"""
环境验证脚本
验证Python环境和依赖是否正确安装
"""

import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType


def verify_environment():
    """验证环境配置"""
    print("="*60)
    print("环境验证")
    print("="*60)
    
    # 1. Python版本
    print(f"✓ Python: {sys.version}")
    
    # 2. PyTorch
    print(f"✓ PyTorch: {torch.__version__}")
    
    # 3. GPU/MPS
    if torch.backends.mps.is_available():
        print(f"✓ MPS (Apple GPU): 可用")
        device = "mps"
    elif torch.cuda.is_available():
        print(f"✓ CUDA GPU: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        print(f"⚠ 使用CPU训练（速度较慢）")
        device = "cpu"
    
    # 4. 其他库
    try:
        import transformers
        import peft
        import scipy
        import numpy as np
        import pandas as pd
        
        print(f"✓ Transformers: {transformers.__version__}")
        print(f"✓ PEFT: {peft.__version__}")
        print(f"✓ SciPy: {scipy.__version__}")
        print(f"✓ NumPy: {np.__version__}")
        print(f"✓ Pandas: {pd.__version__}")
    except ImportError as e:
        print(f"✗ 缺少依赖: {e}")
        return False
    
    # 5. 模型加载测试
    print("\n测试模型加载...")
    try:
        model_name = "MS846/fitness-gpt2-124m"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["c_attn"],
            lora_dropout=0.05,
            bias="none",
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        print("✓ 模型加载与LoRA配置成功")
        
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return False
    
    print("="*60)
    print("环境验证通过！")
    return True


if __name__ == "__main__":
    verify_environment()
