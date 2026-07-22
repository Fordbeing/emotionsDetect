"""
训练脚本 - 使用LoRA微调fitness-gpt2进行情绪识别
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tiktoken
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpt2_model import GPTModel, generate
from lora import apply_lora, save_lora
from huggingface_hub import hf_hub_download


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
    
    def __getitem__(self, idx):
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
        
        full_ids = self.tokenizer.encode(full_text)[:self.max_length]
        prompt_ids = self.tokenizer.encode(prompt_text)[:self.max_length]
        
        # 填充
        padding_length = self.max_length - len(full_ids)
        input_ids = full_ids + [50256] * padding_length  # 50256是GPT-2的pad_token
        
        # 创建标签
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        labels = labels[:self.max_length]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }


def train(args):
    """训练函数"""
    print("="*60)
    print("Fitness-GPT2 LoRA 微调 - 情绪识别")
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
    model.to(device)
    
    # 应用LoRA
    print("\n配置LoRA...")
    model = apply_lora(model, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
    
    # 加载数据
    print("\n加载数据集...")
    tokenizer = tiktoken.get_encoding('gpt2')
    train_dataset = EmotionDataset(args.train_data, tokenizer, args.max_length)
    eval_dataset = EmotionDataset(args.val_data, tokenizer, args.max_length)
    
    print(f"训练样本数: {len(train_dataset)}")
    print(f"验证样本数: {len(eval_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size * 2)
    
    # 优化器
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    
    # 学习率调度
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * 0.1)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.0, 1.0 - (step - warmup_steps) / (total_steps - warmup_steps))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # 交叉熵损失
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    # 训练
    print("\n开始训练...")
    best_eval_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in progress:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            progress.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = total_loss / len(train_loader)
        
        # 验证
        model.eval()
        eval_loss = 0
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                
                logits = model(input_ids)
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                eval_loss += loss.item()
        
        avg_eval_loss = eval_loss / len(eval_loader)
        
        print(f"\nEpoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Eval Loss={avg_eval_loss:.4f}")
        
        # 保存最佳模型
        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            save_path = os.path.join(args.output_dir, 'lora_weights.pth')
            save_lora(model, save_path)
            print(f"  ✓ 保存最佳模型 (Eval Loss: {avg_eval_loss:.4f})")
    
    print("\n" + "="*60)
    print("训练完成！")
    print(f"最佳验证Loss: {best_eval_loss:.4f}")
    print("="*60)
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LoRA微调训练")
    
    parser.add_argument("--train_data", type=str, default="train_split.json")
    parser.add_argument("--val_data", type=str, default="val_split.json")
    parser.add_argument("--output_dir", type=str, default="./models/adapter")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
