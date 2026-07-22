"""
Qwen2.5-7B-Instruct 评估脚本

加载 int4 量化模型 + LoRA adapter，在测试集上评估情绪分类准确率。
"""

import torch
import json
import sys
import os
import re
from pathlib import Path
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


SYSTEM_PROMPT = "You are an emotion classification assistant. Given physiological data, classify the user's emotional state as exactly one word: happy, stressed, sad, or neutral."

VALID_LABELS = {"happy", "stressed", "sad", "neutral"}


def extract_emotion_label(response: str) -> str:
    """从模型输出中提取情绪标签"""
    response_lower = response.lower().strip()

    # 直接匹配
    if response_lower in VALID_LABELS:
        return response_lower

    # 尝试提取第一个有效标签
    for label in VALID_LABELS:
        if label in response_lower:
            return label

    # 无法识别时返回 unknown，避免污染评估指标
    return "unknown"


def load_model(base_model_name, adapter_path, use_4bit=True):
    """加载模型和 LoRA adapter"""
    print(f"加载基座模型: {base_model_name}")

    # MPS 不支持 bitsandbytes 量化
    if use_4bit and torch.backends.mps.is_available():
        print("  注意: MPS 不支持 bitsandbytes int4，自动切换为 fp16")
        use_4bit = False

    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None

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
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )

    # 加载 LoRA adapter
    print(f"加载 LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    return model, tokenizer


def predict_single(model, tokenizer, sample, max_new_tokens=20):
    """单条样本推理"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{sample['instruction']}\n\n{sample['input']}"},
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy for evaluation
            pad_token_id=tokenizer.pad_token_id,
        )

    # 只取新生成的 token
    new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return response


def evaluate(model, tokenizer, test_data, max_samples=None):
    """在测试集上评估"""
    if max_samples:
        test_data = test_data[:max_samples]

    predictions = []
    true_labels = []
    raw_outputs = []

    for i, sample in enumerate(test_data):
        response = predict_single(model, tokenizer, sample)
        pred = extract_emotion_label(response)
        true = sample['output'].strip().lower()

        predictions.append(pred)
        true_labels.append(true)
        raw_outputs.append(response)

        if (i + 1) % 50 == 0:
            print(f"  已评估: {i + 1}/{len(test_data)}")

    return predictions, true_labels, raw_outputs


def calculate_metrics(predictions, true_labels):
    """计算评估指标（排除 unknown 预测）"""
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        accuracy_score,
    )

    # 统计 unknown 数量
    unknown_count = sum(1 for p in predictions if p == "unknown")

    # 过滤掉 unknown 预测
    filtered = [(t, p) for t, p in zip(true_labels, predictions) if p != "unknown"]
    if filtered:
        f_true, f_pred = zip(*filtered)
    else:
        f_true, f_pred = [], []

    accuracy = accuracy_score(f_true, f_pred) if f_true else 0.0

    report = classification_report(
        f_true,
        f_pred,
        labels=["happy", "stressed", "sad", "neutral"],
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(
        f_true,
        f_pred,
        labels=["happy", "stressed", "sad", "neutral"],
    )

    return accuracy, report, cm, unknown_count


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Qwen2.5-7B 情绪识别评估")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter_path", type=str, default="./models/qwen_lora")
    parser.add_argument("--test_data", type=str, default="./data/processed/test_split.json")
    parser.add_argument("--output_file", type=str, default="./evaluation_results_v2.json")
    parser.add_argument("--no_4bit", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    args.use_4bit = not args.no_4bit

    print("=" * 60)
    print("Qwen2.5-7B 情绪识别评估")
    print("=" * 60)

    # 加载模型
    model, tokenizer = load_model(args.base_model, args.adapter_path, args.use_4bit)

    # 加载测试数据
    print(f"\n加载测试数据: {args.test_data}")
    with open(args.test_data, 'r') as f:
        test_data = json.load(f)
    print(f"测试样本数: {len(test_data)}")

    # 评估
    print("\n开始评估...")
    predictions, true_labels, raw_outputs = evaluate(
        model, tokenizer, test_data, args.max_samples
    )

    # 计算指标
    accuracy, report, cm, unknown_count = calculate_metrics(predictions, true_labels)

    # 打印结果
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)

    if unknown_count > 0:
        print(f"\n[注意] {unknown_count} 条预测无法识别标签（已排除）")

    print(f"\n整体准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    print("\n各类别详细指标:")
    print("-" * 50)
    for label in ["happy", "stressed", "sad", "neutral"]:
        if label in report:
            m = report[label]
            print(f"\n{label}:")
            print(f"  Precision: {m['precision']:.4f}")
            print(f"  Recall:    {m['recall']:.4f}")
            print(f"  F1-Score:  {m['f1-score']:.4f}")
            print(f"  Support:   {m['support']}")

    print(f"\nMacro Avg F1:    {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted Avg F1: {report['weighted avg']['f1-score']:.4f}")

    print("\n混淆矩阵:")
    print("         pred: happy  stressed  sad  neutral")
    labels = ["happy", "stressed", "sad", "neutral"]
    for i, row in enumerate(cm):
        print(f"true {labels[i]:>10}: {row}")

    # 错误分析
    print("\n" + "=" * 60)
    print("错误分析")
    print("=" * 60)

    errors = [(t, p, r) for t, p, r in zip(true_labels, predictions, raw_outputs) if t != p]
    print(f"错误样本数: {len(errors)} / {len(true_labels)}")

    error_counts = Counter([(t, p) for t, p, _ in errors])
    print("\n最常见错误:")
    for (true, pred), count in error_counts.most_common(5):
        print(f"  真实: {true} → 预测: {pred} ({count}次)")

    # 展示一些错误样本
    if errors:
        print("\n错误样本示例 (前5个):")
        for true, pred, raw in errors[:5]:
            print(f"  真实={true}, 预测={pred}, 模型输出='{raw}'")

    # 保存结果
    results = {
        "accuracy": accuracy,
        "macro_f1": report['macro avg']['f1-score'],
        "weighted_f1": report['weighted avg']['f1-score'],
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "predictions": predictions,
        "true_labels": true_labels,
        "raw_outputs": raw_outputs,
    }

    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {args.output_file}")

    return accuracy, report


if __name__ == "__main__":
    main()
