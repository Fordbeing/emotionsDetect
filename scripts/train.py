"""
训练脚本
使用LoRA微调Fitness-GPT2进行情绪识别
"""

import os
import sys
import torch
import json
from torch.utils.data import Dataset
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from pathlib import Path
import argparse
from typing import Dict

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_config import load_model_with_lora


class EmotionDataset(Dataset):
    """情绪识别数据集"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 256):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.prompt_template = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n"
            "### Input:\n{input}\n\n"
            "### Response:\n{output}"
        )
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        
        full_text = self.prompt_template.format(
            instruction=sample['instruction'],
            input=sample['input'],
            output=sample['output']
        )
        
        prompt_text = self.prompt_template.format(
            instruction=sample['instruction'],
            input=sample['input'],
            output=""
        )
        
        full_encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        prompt_encoding = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        input_ids = full_encoding['input_ids'].squeeze()
        labels = input_ids.clone()
        
        prompt_len = prompt_encoding['attention_mask'].sum().item()
        labels[:prompt_len] = -100
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': full_encoding['attention_mask'].squeeze(),
            'labels': labels
        }


def train(args):
    """主训练函数"""
    print("="*60)
    print("Fitness-GPT2 LoRA 微调 - 情绪识别")
    print("="*60)
    
    # 1. 加载模型
    model, tokenizer = load_model_with_lora(
        model_name=args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    
    # 2. 加载数据
    print("\n加载数据集...")
    train_dataset = EmotionDataset(args.train_data, tokenizer, args.max_length)
    eval_dataset = EmotionDataset(args.val_data, tokenizer, args.max_length)
    
    print(f"训练样本数: {len(train_dataset)}")
    print(f"验证样本数: {len(eval_dataset)}")
    
    # 3. 配置训练参数
    # 根据设备调整batch_size
    if torch.backends.mps.is_available() or torch.cuda.is_available():
        batch_size = 8
    else:
        batch_size = 4
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=2,
        optim="adamw_torch",
        learning_rate=args.lr,
        weight_decay=0.01,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=torch.backends.mps.is_available() or torch.cuda.is_available(),
        dataloader_num_workers=0,
        report_to="none",
        seed=42,
    )
    
    # 4. 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # 5. 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=5)
        ]
    )
    
    # 6. 开始训练
    print("\n开始训练...")
    train_result = trainer.train()
    
    # 7. 保存模型
    print("\n保存模型...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # 保存训练结果
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    print(f"\n训练完成！模型保存在: {args.output_dir}")
    
    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA微调训练")
    
    parser.add_argument("--model_name", type=str, default="MS846/fitness-gpt2-124m")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--train_data", type=str, default="train_split.json")
    parser.add_argument("--val_data", type=str, default="val_split.json")
    parser.add_argument("--output_dir", type=str, default="./models/adapter")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=100)
    
    args = parser.parse_args()
    train(args)
