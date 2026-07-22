"""
评估脚本
评估训练好的LoRA模型在测试集上的表现
"""

import torch
import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.metrics import classification_report, confusion_matrix
from typing import List, Dict
from tqdm import tqdm
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class EmotionEvaluator:
    """情绪识别评估器"""
    
    def __init__(self, base_model_name: str, adapter_path: str):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        
        print("加载模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"模型加载完成，设备: {self.device}")
    
    @torch.no_grad()
    def predict(self, instruction: str, input_text: str, max_new_tokens: int = 100) -> str:
        """单条预测"""
        prompt = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n"
            "### Input:\n{input}\n\n"
            "### Response:\n"
        ).format(instruction=instruction, input=input_text)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=3,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "### Response:\n" in response:
            response = response.split("### Response:\n")[-1].strip()
        
        return response
    
    def extract_emotion_label(self, response: str) -> str:
        """从输出中提取情绪标签"""
        response_lower = response.lower()
        
        emotion_keywords = {
            "happy/excited": ["happy", "excited", "positive", "high energy"],
            "stressed/anxious": ["stressed", "anxious", "tense", "nervous"],
            "sad/depressed": ["sad", "depressed", "negative", "low energy"],
            "neutral/calm": ["neutral", "calm", "relaxed", "stable"]
        }
        
        for emotion, keywords in emotion_keywords.items():
            if any(keyword in response_lower for keyword in keywords):
                return emotion
        
        return "neutral/calm"
    
    def evaluate_dataset(self, test_data: List[Dict]) -> Dict:
        """在测试集上评估"""
        predictions = []
        true_labels = []
        
        print("\n开始评估...")
        for sample in tqdm(test_data, desc="评估进度"):
            response = self.predict(
                instruction=sample['instruction'],
                input_text=sample['input']
            )
            
            pred_emotion = self.extract_emotion_label(response)
            true_emotion = sample['output'].split('.')[0].strip()
            
            predictions.append(pred_emotion)
            true_labels.append(true_emotion)
        
        report = classification_report(
            true_labels, 
            predictions, 
            output_dict=True,
            zero_division=0
        )
        
        cm = confusion_matrix(true_labels, predictions, 
                             labels=["happy/excited", "stressed/anxious", 
                                     "sad/depressed", "neutral/calm"])
        
        return {
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "predictions": predictions,
            "true_labels": true_labels
        }
    
    def print_results(self, results: Dict):
        """打印评估结果"""
        print("\n" + "="*60)
        print("评估结果")
        print("="*60)
        
        report = results["classification_report"]
        
        print(f"\n整体准确率: {report['accuracy']:.4f}")
        print(f"Macro F1: {report['macro avg']['f1-score']:.4f}")
        print(f"Weighted F1: {report['weighted avg']['f1-score']:.4f}")
        
        print("\n各类别详细指标:")
        print("-"*50)
        for emotion in ["happy/excited", "stressed/anxious", "sad/depressed", "neutral/calm"]:
            if emotion in report:
                metrics = report[emotion]
                print(f"{emotion}:")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall: {metrics['recall']:.4f}")
                print(f"  F1: {metrics['f1-score']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="MS846/fitness-gpt2-124m")
    parser.add_argument("--adapter_path", type=str, default="./models/adapter")
    parser.add_argument("--test_data", type=str, default="test_split.json")
    parser.add_argument("--output", type=str, default="evaluation_results.json")
    
    args = parser.parse_args()
    
    with open(args.test_data, 'r') as f:
        test_data = json.load(f)
    
    evaluator = EmotionEvaluator(args.base_model, args.adapter_path)
    results = evaluator.evaluate_dataset(test_data)
    evaluator.print_results(results)
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n评估结果已保存到: {args.output}")
