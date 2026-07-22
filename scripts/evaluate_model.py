"""
模型评估脚本
测试模型在测试集上的准确率和分类效果
"""

import torch
import tiktoken
import json
import sys
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpt2_model import GPTModel, generate
from lora import load_lora
from huggingface_hub import hf_hub_download


def extract_emotion_label(response):
    """从模型输出中提取情绪标签"""
    response_lower = response.lower()
    
    emotion_keywords = {
        "happy/excited": ["happy", "excited", "positive", "high energy", "good"],
        "stressed/anxious": ["stressed", "anxious", "tense", "nervous", "bad"],
        "sad/depressed": ["sad", "depressed", "negative", "low energy", "tired"],
        "neutral/calm": ["neutral", "calm", "relaxed", "stable", "normal"]
    }
    
    for emotion, keywords in emotion_keywords.items():
        if any(keyword in response_lower for keyword in keywords):
            return emotion
    
    # 默认返回
    if any(word in response_lower for word in ["safe", "healthy", "well"]):
        return "neutral/calm"
    return "neutral/calm"


def evaluate_model(test_data, model, tokenizer, device, max_samples=None):
    """评估模型"""
    predictions = []
    true_labels = []
    
    if max_samples:
        test_data = test_data[:max_samples]
    
    for i, sample in enumerate(test_data):
        prompt = f'''Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{sample['instruction']}

### Input:
{sample['input']}

### Response:
'''
        
        ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            out_ids = generate(
                model=model, 
                idx=ids, 
                max_new_tokens=100, 
                context_size=1024, 
                eos_id=50256, 
                top_k=50, 
                temperature=0.7
            )
        
        response = tokenizer.decode(out_ids[0].tolist())[len(prompt):]
        pred_emotion = extract_emotion_label(response)
        
        # 从输出中提取真实标签
        true_emotion = sample['output'].split('.')[0].strip().replace('The user is likely feeling ', '')
        
        predictions.append(pred_emotion)
        true_labels.append(true_emotion)
        
        if (i + 1) % 10 == 0:
            print(f"已评估: {i+1}/{len(test_data)}")
    
    return predictions, true_labels


def calculate_metrics(predictions, true_labels):
    """计算评估指标"""
    accuracy = accuracy_score(true_labels, predictions)
    
    report = classification_report(
        true_labels, 
        predictions, 
        output_dict=True,
        zero_division=0
    )
    
    cm = confusion_matrix(
        true_labels, 
        predictions,
        labels=["happy/excited", "stressed/anxious", "sad/depressed", "neutral/calm"]
    )
    
    return accuracy, report, cm


def main():
    print("="*60)
    print("Fitness-GPT2 LoRA 模型评估")
    print("="*60)
    
    # 设备
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"设备: {device}")
    
    # 加载模型
    print("\n加载模型...")
    BASE_CONFIG = {
        'vocab_size': 50257, 'context_length': 1024, 'drop_rate': 0.0,
        'qkv_bias': True, 'emb_dim': 768, 'n_layers': 12, 'n_heads': 12,
    }
    
    ckpt_path = hf_hub_download(repo_id='MS846/fitness-gpt2-124m', filename='model_fitness_small.pth')
    
    model = GPTModel(BASE_CONFIG)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    
    # 加载LoRA
    lora_path = '/Users/wrf/Desktop/fanxing/models/adapter/lora_weights.pth'
    model = load_lora(model, lora_path)
    model.to(device).eval()
    
    print(f"✓ 模型加载成功")
    
    # 加载测试数据
    print("\n加载测试数据...")
    with open('/Users/wrf/Desktop/fanxing/data/processed/test_split.json', 'r') as f:
        test_data = json.load(f)
    print(f"✓ 测试样本数: {len(test_data)}")
    
    # 加载tokenizer
    tokenizer = tiktoken.get_encoding('gpt2')
    
    # 运行评估
    print("\n开始评估...")
    predictions, true_labels = evaluate_model(test_data, model, tokenizer, device, max_samples=50)
    
    # 计算指标
    accuracy, report, cm = calculate_metrics(predictions, true_labels)
    
    # 打印结果
    print("\n" + "="*60)
    print("评估结果")
    print("="*60)
    
    print(f"\n整体准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\n各类别详细指标:")
    print("-"*50)
    for emotion in ["happy/excited", "stressed/anxious", "sad/depressed", "neutral/calm"]:
        if emotion in report:
            metrics = report[emotion]
            print(f"\n{emotion}:")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-Score: {metrics['f1-score']:.4f}")
            print(f"  Support: {metrics['support']}")
    
    print(f"\nMacro Avg F1: {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted Avg F1: {report['weighted avg']['f1-score']:.4f}")
    
    print("\n混淆矩阵:")
    print(cm)
    
    # 保存结果
    results = {
        "accuracy": accuracy,
        "macro_f1": report['macro avg']['f1-score'],
        "weighted_f1": report['weighted avg']['f1-score'],
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "predictions": predictions,
        "true_labels": true_labels
    }
    
    with open('/Users/wrf/Desktop/fanxing/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ 评估结果已保存到: evaluation_results.json")
    
    # 分析错误
    print("\n" + "="*60)
    print("错误分析")
    print("="*60)
    
    errors = [(t, p) for t, p in zip(true_labels, predictions) if t != p]
    print(f"错误样本数: {len(errors)} / {len(true_labels)}")
    
    error_counts = Counter(errors)
    print("\n最常见错误:")
    for (true, pred), count in error_counts.most_common(5):
        print(f"  真实: {true} → 预测: {pred} ({count}次)")
    
    return accuracy, report


if __name__ == "__main__":
    main()
