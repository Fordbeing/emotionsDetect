"""
模型配置模块
加载fitness-gpt2并配置LoRA
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from typing import Tuple, Optional


def load_model_with_lora(
    model_name: str = "MS846/fitness-gpt2-124m",
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    adapter_path: Optional[str] = None
) -> Tuple:
    """
    加载模型并配置LoRA
    
    Returns:
        model, tokenizer
    """
    print(f"加载分词器: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    print(f"加载模型: {model_name}")
    
    # 检查设备
    if torch.backends.mps.is_available():
        device_map = "mps"
        torch_dtype = torch.float16
    elif torch.cuda.is_available():
        device_map = "auto"
        torch_dtype = torch.float16
    else:
        device_map = "cpu"
        torch_dtype = torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )
    
    # 如果有已训练的适配器，加载它
    if adapter_path:
        print(f"加载LoRA适配器: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    else:
        # 配置新的LoRA
        print("配置新的LoRA...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["c_attn"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
    
    model.print_trainable_parameters()
    
    return model, tokenizer


def get_model_info(model) -> dict:
    """获取模型信息"""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    
    return {
        "trainable_parameters": trainable_params,
        "total_parameters": all_params,
        "trainable_percentage": f"{100 * trainable_params / all_params:.2f}%",
    }


if __name__ == "__main__":
    print("="*50)
    print("Fitness-GPT2 + LoRA 模型配置测试")
    print("="*50)
    
    model, tokenizer = load_model_with_lora()
    info = get_model_info(model)
    
    print(f"\n可训练参数: {info['trainable_parameters']:,}")
    print(f"总参数: {info['total_parameters']:,}")
    print(f"可训练比例: {info['trainable_percentage']}")
    
    # 测试推理
    test_input = "Heart Rate: 85 bpm. Heart Rate Variability (RMSSD): 30 ms."
    print(f"\n测试输入: {test_input}")
