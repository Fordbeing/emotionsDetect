"""
模型保存验证脚本

验证 LoRA adapter 保存完整、可加载、可推理。
"""

import torch
import json
import sys
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def verify_adapter(adapter_path):
    """验证 adapter 文件完整性"""
    adapter_path = Path(adapter_path)
    print(f"验证 adapter 目录: {adapter_path}")

    required_files = [
        "adapter_config.json",
        "adapter_model.safetensors",
    ]

    # 也检查 .bin 格式
    alt_files = [
        ("adapter_model.safetensors", "adapter_model.bin"),
    ]

    all_ok = True

    for f in required_files:
        path = adapter_path / f
        if path.exists():
            size = path.stat().st_size
            print(f"  {f}: OK ({size / 1024:.1f} KB)")
        else:
            # 检查替代文件
            alt = dict(alt_files).get(f)
            if alt and (adapter_path / alt).exists():
                size = (adapter_path / alt).stat().st_size
                print(f"  {alt}: OK ({size / 1024:.1f} KB)")
            else:
                print(f"  {f}: MISSING!")
                all_ok = False

    # 检查 adapter_config.json 内容
    config_path = adapter_path / "adapter_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        print(f"\nAdapter 配置:")
        print(f"  LoRA r: {config.get('r')}")
        print(f"  LoRA alpha: {config.get('lora_alpha')}")
        print(f"  Target modules: {config.get('target_modules')}")
        print(f"  Base model: {config.get('base_model_name_or_path')}")

    return all_ok


def verify_loading(base_model_name, adapter_path):
    """验证模型加载和推理"""
    print(f"\n加载基座模型: {base_model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    is_mps = torch.backends.mps.is_available()

    if is_mps:
        print("  加载到 CPU 后移至 MPS...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="cpu",
        )
        model = model.to("mps")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )

    print(f"加载 adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    # 测试推理
    test_input = "Heart Rate: 95 bpm. RMSSD: 18 ms. SDNN: 22 ms. PNN50: 8%."
    messages = [
        {"role": "system", "content": "You are an emotion classification assistant. Given physiological data, classify the user's emotional state as exactly one word: happy, stressed, sad, or neutral."},
        {"role": "user", "content": f"Based on the following physiological data, classify the user's emotional state as one of: happy, stressed, sad, neutral.\n\n{test_input}"},
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    print(f"\n测试推理:")
    print(f"  输入: {test_input}")
    print(f"  输出: {response}")

    valid_labels = {"happy", "stressed", "sad", "neutral"}
    if response.lower().strip() in valid_labels:
        print(f"  状态: OK (有效标签)")
        return True
    else:
        # 检查是否包含有效标签
        for label in valid_labels:
            if label in response.lower():
                print(f"  状态: OK (包含有效标签 '{label}')")
                return True
        print(f"  状态: WARNING (输出不在预期标签中)")
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="验证模型保存")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter_path", type=str, default="./models/qwen_lora")
    parser.add_argument("--skip_inference", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("模型保存验证")
    print("=" * 60)

    # 1. 验证文件完整性
    files_ok = verify_adapter(args.adapter_path)

    if not files_ok:
        print("\n验证失败：adapter 文件不完整")
        sys.exit(1)

    # 2. 验证加载和推理
    if not args.skip_inference:
        inference_ok = verify_loading(args.base_model, args.adapter_path)
    else:
        inference_ok = True
        print("\n跳过推理验证")

    # 总结
    print("\n" + "=" * 60)
    if files_ok and inference_ok:
        print("验证通过！模型保存完整且可正常使用。")
    else:
        print("验证有警告，请检查上方输出。")
    print("=" * 60)


if __name__ == "__main__":
    main()
