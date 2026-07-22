# Fitness-GPT2 + LoRA 情绪识别微调：深度执行手册

> 版本：2.0 深化版 | 最后更新：2026-07-20

---

## 目录

1. [项目概述与技术路线](#1-项目概述与技术路线)
2. [环境搭建与验证](#2-环境搭建与验证)
3. [数据获取与处理（深度版）](#3-数据获取与处理深度版)
4. [模型配置与LoRA适配](#4-模型配置与lora适配)
5. [训练流程优化](#5-训练流程优化)
6. [评估与推理](#6-评估与推理)
7. [故障排除与调试](#7-故障排除与调试)
8. [进阶优化策略](#8-进阶优化策略)
9. [参考资源](#9-参考资源)

---

## 1. 项目概述与技术路线

### 1.1 核心思路

```
心率信号 (HR/HRV) → 信号处理 → 文本描述 → Fitness-GPT2 + LoRA → 情绪标签
```

### 1.2 技术选型理由

| 组件 | 选择 | 理由 |
|------|------|------|
| 基座模型 | MS846/fitness-gpt2-124m | 已针对生理信号预训练，理解HRV特征 |
| 微调方法 | LoRA (r=8, alpha=16) | 仅训练0.25%参数，避免灾难性遗忘 |
| 数据集 | DEAP/AMIGOS | 学术标准数据集，含HR和情绪标注 |
| 评估指标 | Macro F1 + Loss | 处理类别不平衡 |

### 1.3 预期成果

- 训练完成的LoRA适配器 (~88MB)
- 支持实时推理的API接口
- 完整的评估报告

---

## 2. 环境搭建与验证

### 2.1 完整依赖安装

```bash
# 创建环境
conda create -n lora_emotion python=3.10 -y
conda activate lora_emotion

# 核心依赖
pip install torch>=2.0 transformers>=4.35 peft>=0.6 datasets accelerate

# 数据处理
pip install scipy numpy pandas scikit-learn

# 信号处理 (用于HRV特征提取)
pip install neurokit2 heartpy

# 可视化 (可选)
pip install matplotlib seaborn

# 评估增强
pip install evaluate rouge_score
```

### 2.2 环境验证脚本

创建 `verify_env.py`:

```python
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

def verify_environment():
    print("=" * 50)
    print("环境验证")
    print("=" * 50)
    
    # 1. Python版本
    print(f"✓ Python: {sys.version}")
    
    # 2. PyTorch与GPU
    print(f"✓ PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU显存: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        print("⚠ 未检测到GPU，训练将使用CPU（非常慢）")
    
    # 3. 模型加载测试
    print("\n测试模型加载...")
    try:
        model_name = "MS846/fitness-gpt2-124m"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # 4. LoRA配置测试
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
    
    print("=" * 50)
    print("环境验证通过！")
    return True

if __name__ == "__main__":
    verify_environment()
```

运行验证：
```bash
python verify_env.py
```

---

## 3. 数据获取与处理（深度版）

### 3.1 数据集获取

#### DEAP 数据集
- **下载地址**: https://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html
- **格式**: MATLAB (.mat) 文件
- **包含**: 32名被试，40个试次，32通道生理信号 + 4维情绪标签 (Valence, Arousal, Dominance, Liking)

#### AMIGOS 数据集
- **下载地址**: http://www.eecs.qmul.ac.uk/mmv/datasets/amigos/download.html
- **格式**: MATLAB (.mat) 文件
- **包含**: 40名被试，单人/多人观看视频，含ECG（可提取HR）

### 3.2 数据预处理流程

创建 `data_pipeline.py`:

```python
import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import loadmat
from pathlib import Path
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class HRDataProcessor:
    """心率数据处理器：从原始生理信号提取HR/HRV特征"""
    
    # 情绪标签映射（基于DEAP论文）
    EMOTION_MAP = {
        (0, 0): "neutral/calm",      # Low V, Low A
        (0, 1): "stressed/anxious",  # Low V, High A
        (1, 0): "sad/depressed",     # High V, Low A
        (1, 1): "happy/excited",     # High V, High A
    }
    
    def __init__(self, sampling_rate: int = 128):
        self.sr = sampling_rate
        
    def extract_hr_from_ecg(self, ecg_signal: np.ndarray) -> float:
        """从ECG信号提取心率"""
        # 带通滤波 (0.5-40 Hz)
        b, a = signal.butter(4, [0.5, 40], btype='bandpass', fs=self.sr)
        ecg_filtered = signal.filtfilt(b, a, ecg_signal)
        
        # R波检测 (简化版：使用峰值检测)
        peaks, _ = signal.find_peaks(
            ecg_filtered, 
            height=np.std(ecg_filtered) * 0.5,
            distance=int(0.5 * self.sr)  # 最小500ms间隔
        )
        
        if len(peaks) < 2:
            return 0.0
        
        # 计算瞬时心率
        rr_intervals = np.diff(peaks) / self.sr
        hr = 60.0 / np.mean(rr_intervals)
        
        return hr
    
    def compute_hrv_features(self, ecg_signal: np.ndarray) -> Dict[str, float]:
        """计算HRV时域和频域特征"""
        # 提取RR间期
        b, a = signal.butter(4, [0.5, 40], btype='bandpass', fs=self.sr)
        ecg_filtered = signal.filtfilt(b, a, ecg_signal)
        
        peaks, _ = signal.find_peaks(
            ecg_filtered,
            height=np.std(ecg_filtered) * 0.5,
            distance=int(0.5 * self.sr)
        )
        
        if len(peaks) < 10:
            return self._empty_hrv_features()
        
        rr_intervals = np.diff(peaks) / self.sr * 1000  # 转换为ms
        
        features = {}
        
        # 时域特征
        features['rmssd'] = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
        features['sdnn'] = np.std(rr_intervals)
        features['mean_rr'] = np.mean(rr_intervals)
        features['hr'] = 60000 / features['mean_rr']  # 心率 bpm
        features['pnn50'] = np.sum(np.abs(np.diff(rr_intervals)) > 50) / len(rr_intervals) * 100
        
        return features
    
    def _empty_hrv_features(self) -> Dict[str, float]:
        return {
            'rmssd': 0, 'sdnn': 0, 'mean_rr': 0, 
            'hr': 0, 'pnn50': 0
        }
    
    def get_emotion_label(self, valence: float, arousal: float) -> str:
        """将连续的情绪值转换为离散标签"""
        v_class = 1 if valence >= 5 else 0
        a_class = 1 if arousal >= 5 else 0
        return self.EMOTION_MAP[(v_class, a_class)]


def load_deap_dataset(data_dir: str) -> List[Dict]:
    """加载DEAP数据集并提取特征"""
    processor = HRDataProcessor(sampling_rate=128)
    data_path = Path(data_dir)
    
    all_samples = []
    
    for subject_file in sorted(data_path.glob("s*.mat")):
        print(f"处理: {subject_file.name}")
        
        # 加载MAT文件
        mat_data = loadmat(subject_file)
        
        # DEAP数据结构: data[40, 40, 8064] = [trials, channels, samples]
        # channels: 0-31: 生理信号, 32-35: 情绪标签 (V, A, D, L)
        data = mat_data['data']
        
        for trial in range(40):
            # 提取ECG (通道1，索引1，或根据实际情况调整)
            ecg_signal = data[trial, 1, :]  # 调整通道索引
            
            # 提取特征
            hr = processor.extract_hr_from_ecg(ecg_signal)
            hrv_features = processor.compute_hrv_features(ecg_signal)
            
            # 情绪标签 (中值分割为高/低)
            valence = data[trial, 32, 0]
            arousal = data[trial, 33, 0]
            emotion = processor.get_emotion_label(valence, arousal)
            
            sample = {
                'subject': subject_file.stem,
                'trial': trial,
                'hr': hr,
                'hrv': hrv_features,
                'valence': valence,
                'arousal': arousal,
                'emotion': emotion
            }
            
            all_samples.append(sample)
    
    print(f"共提取 {len(all_samples)} 个样本")
    return all_samples


def create_alpaca_format(samples: List[Dict], output_path: str):
    """将特征转换为Alpaca格式的JSON文件"""
    
    alpaca_data = []
    
    INSTRUCTION_TEMPLATE = (
        "Based on the following physiological data, "
        "analyze the user's emotional state."
    )
    
    INPUT_TEMPLATE = (
        "Heart Rate: {hr:.1f} bpm. "
        "Heart Rate Variability (RMSSD): {rmssd:.1f} ms. "
        "SDNN: {sdnn:.1f} ms. "
        "PNN50: {pnn50:.1f}%."
    )
    
    for sample in samples:
        if sample['hr'] == 0:  # 跳过无效数据
            continue
        
        input_text = INPUT_TEMPLATE.format(
            hr=sample['hr'],
            rmssd=sample['hrv']['rmssd'],
            sdnn=sample['hrv']['sdnn'],
            pnn50=sample['hrv']['pnn50']
        )
        
        # 构建更丰富的输出描述
        emotion = sample['emotion']
        valence = sample['valence']
        arousal = sample['arousal']
        
        output_text = (
            f"The user is likely feeling {emotion}. "
            f"Valence level: {valence:.1f}/9, Arousal level: {arousal:.1f}/9. "
            f"This suggests {'positive' if valence >= 5 else 'negative'} "
            f"and {'high energy' if arousal >= 5 else 'low energy'} emotional state."
        )
        
        alpaca_sample = {
            "instruction": INSTRUCTION_TEMPLATE,
            "input": input_text,
            "output": output_text
        }
        
        alpaca_data.append(alpaca_sample)
    
    # 保存为JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(alpaca_data, f, indent=2, ensure_ascii=False)
    
    print(f"保存到 {output_path}: {len(alpaca_data)} 条训练数据")
    return alpaca_data


if __name__ == "__main__":
    # 使用示例
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='DEAP数据目录')
    parser.add_argument('--output', type=str, default='train_data.json', help='输出文件')
    
    args = parser.parse_args()
    
    # 处理数据
    samples = load_deap_dataset(args.data_dir)
    
    # 创建Alpaca格式
    create_alpaca_format(samples, args.output)
```

### 3.3 数据划分

```python
from sklearn.model_selection import train_test_split

def split_dataset(data_path: str, test_size: float = 0.15, val_size: float = 0.15):
    """划分训练/验证/测试集"""
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # 第一次划分：分离测试集
    train_val, test = train_test_split(data, test_size=test_size, random_state=42, 
                                       stratify=[d['output'].split('.')[0] for d in data])
    
    # 第二次划分：分离验证集
    relative_val_size = val_size / (1 - test_size)
    train, val = train_test_split(train_val, test_size=relative_val_size, random_state=42,
                                  stratify=[d['output'].split('.')[0] for d in train_val])
    
    # 保存
    for name, dataset in [('train', train), ('val', val), ('test', test)]:
        with open(f'{name}_split.json', 'w') as f:
            json.dump(dataset, f, indent=2)
    
    print(f"训练集: {len(train)} | 验证集: {len(val)} | 测试集: {len(test)}")
    
    return train, val, test
```

---

## 4. 模型配置与LoRA适配

### 4.1 完整的模型加载与配置

创建 `model_config.py`:

```python
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType,
    prepare_model_for_kbit_training
)
from typing import Optional

def load_model_with_lora(
    model_name: str = "MS846/fitness-gpt2-124m",
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    use_4bit: bool = False,
    device_map: str = "auto"
):
    """
    加载模型并配置LoRA
    
    Args:
        model_name: HuggingFace模型名称
        lora_r: LoRA秩（越大越强大，但参数越多）
        lora_alpha: 缩放因子（通常为2*r）
        lora_dropout: Dropout率
        use_4bit: 是否使用4bit量化（节省显存）
        device_map: 设备映射策略
    
    Returns:
        model, tokenizer
    """
    
    # 1. 加载分词器
    print(f"加载分词器: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # 2. 量化配置（可选）
    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    
    # 3. 加载模型
    print(f"加载模型: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    
    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    
    # 4. 配置LoRA
    print("配置LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["c_attn"],  # GPT-2的注意力层
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    
    # 打印参数统计
    model.print_trainable_parameters()
    
    return model, tokenizer


def get_model_info(model):
    """获取模型详细信息"""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    
    info = {
        "trainable_parameters": trainable_params,
        "total_parameters": all_params,
        "trainable_percentage": f"{100 * trainable_params / all_params:.2f}%",
        "model_memory_mb": f"{all_params * 4 / 1024**2:.1f}",  # 假设float32
    }
    
    return info
```

### 4.2 LoRA参数选择指南

| 场景 | r | alpha | dropout | 说明 |
|------|---|-------|---------|------|
| 快速实验 | 4 | 8 | 0.1 | 最少参数，最快训练 |
| **推荐配置** | **8** | **16** | **0.05** | **性能与效率平衡** |
| 高精度 | 16 | 32 | 0.1 | 更强表达力，参数翻倍 |
| 极端精度 | 32 | 64 | 0.15 | 仅用于大模型+大数据 |

---

## 5. 训练流程优化

### 5.1 数据分词与格式化

创建 `data_collator.py`:

```python
import torch
from torch.utils.data import Dataset
from typing import Dict, List
import json

class EmotionDataset(Dataset):
    """情绪识别数据集"""
    
    def __init__(
        self, 
        data_path: str, 
        tokenizer, 
        max_length: int = 256,
        prompt_template: str = None
    ):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 默认prompt模板
        self.prompt_template = prompt_template or (
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
        
        # 构建完整文本
        full_text = self.prompt_template.format(
            instruction=sample['instruction'],
            input=sample['input'],
            output=sample['output']
        )
        
        # 构建仅prompt部分（用于计算标签）
        prompt_text = self.prompt_template.format(
            instruction=sample['instruction'],
            input=sample['input'],
            output=""
        )
        
        # 分词
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
        
        # 创建标签（只计算output部分的loss）
        input_ids = full_encoding['input_ids'].squeeze()
        labels = input_ids.clone()
        
        # 将prompt部分的标签设为-100（忽略）
        prompt_len = prompt_encoding['attention_mask'].sum().item()
        labels[:prompt_len] = -100
        
        # 将padding部分的标签设为-100
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': full_encoding['attention_mask'].squeeze(),
            'labels': labels
        }
```

### 5.2 完整的训练脚本

创建 `train.py`:

```python
import os
import torch
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from datasets import load_dataset
import json
import argparse
from pathlib import Path

# 导入自定义模块
from model_config import load_model_with_lora
from data_collator import EmotionDataset

def setup_training_args(args) -> TrainingArguments:
    """配置训练参数"""
    
    # 根据GPU显存自动调整batch_size
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_mem / 1e9
        if gpu_memory < 8:
            batch_size = 4
            gradient_accum = 4
        elif gpu_memory < 16:
            batch_size = 8
            gradient_accum = 2
        else:
            batch_size = 16
            gradient_accum = 1
    else:
        batch_size = 2
        gradient_accum = 8
    
    training_args = TrainingArguments(
        # 输出目录
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        
        # 训练轮数
        num_train_epochs=args.epochs,
        
        # 批次大小
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=gradient_accum,
        
        # 优化器
        optim="adamw_torch",
        learning_rate=args.lr,
        weight_decay=0.01,
        max_grad_norm=1.0,
        
        # 学习率调度
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        
        # 日志与保存
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        
        # 评估
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        
        # 性能优化
        fp16=torch.cuda.is_available(),
        bf16=False,
        dataloader_num_workers=4,
        group_by_length=True,
        
        # 其他
        report_to="none",
        seed=42,
    )
    
    return training_args


def train(args):
    """主训练流程"""
    
    print("=" * 60)
    print("Fitness-GPT2 LoRA 微调 - 情绪识别")
    print("=" * 60)
    
    # 1. 加载模型
    model, tokenizer = load_model_with_lora(
        model_name=args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_4bit=args.use_4bit
    )
    
    # 2. 加载数据
    print("\n加载数据集...")
    train_dataset = EmotionDataset(
        data_path=args.train_data,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    eval_dataset = EmotionDataset(
        data_path=args.val_data,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    print(f"训练样本数: {len(train_dataset)}")
    print(f"验证样本数: {len(eval_dataset)}")
    
    # 3. 配置训练
    training_args = setup_training_args(args)
    
    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # 因果语言模型
    )
    
    # 4. 创建Trainer
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
    
    # 5. 开始训练
    print("\n开始训练...")
    train_result = trainer.train()
    
    # 6. 保存模型
    print("\n保存模型...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # 保存训练结果
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    print(f"\n训练完成！模型保存在: {args.output_dir}")
    
    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA微调训练")
    
    # 模型参数
    parser.add_argument("--model_name", type=str, default="MS846/fitness-gpt2-124m")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--use_4bit", action="store_true")
    
    # 数据参数
    parser.add_argument("--train_data", type=str, default="train_split.json")
    parser.add_argument("--val_data", type=str, default="val_split.json")
    parser.add_argument("--max_length", type=int, default=256)
    
    # 训练参数
    parser.add_argument("--output_dir", type=str, default="./emotion-lora-adapter")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=100)
    
    args = parser.parse_args()
    
    train(args)
```

### 5.3 训练启动命令

```bash
# 基础训练
python train.py \
    --train_data train_split.json \
    --val_data val_split.json \
    --output_dir ./emotion-lora-adapter \
    --epochs 3 \
    --lr 5e-5

# 使用4bit量化（节省显存）
python train.py \
    --train_data train_split.json \
    --val_data val_split.json \
    --output_dir ./emotion-lora-adapter \
    --epochs 3 \
    --lr 5e-5 \
    --use_4bit

# 自定义LoRA参数
python train.py \
    --train_data train_split.json \
    --val_data val_split.json \
    --output_dir ./emotion-lora-adapter \
    --epochs 5 \
    --lr 3e-5 \
    --lora_r 16 \
    --lora_alpha 32
```

---

## 6. 评估与推理

### 6.1 评估脚本

创建 `evaluate.py`:

```python
import torch
import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.metrics import classification_report, confusion_matrix
import argparse
from typing import List, Dict
from tqdm import tqdm

class EmotionEvaluator:
    """情绪识别评估器"""
    
    def __init__(self, base_model_name: str, adapter_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 加载模型
        print("加载模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"模型加载完成，使用设备: {self.device}")
    
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
        
        # 提取Response部分
        if "### Response:\n" in response:
            response = response.split("### Response:\n")[-1].strip()
        
        return response
    
    def extract_emotion_label(self, response: str) -> str:
        """从模型输出中提取情绪标签"""
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
        
        return "neutral/calm"  # 默认
    
    def evaluate_dataset(self, test_data: List[Dict]) -> Dict:
        """在测试集上评估"""
        predictions = []
        true_labels = []
        
        print("\n开始评估...")
        for sample in tqdm(test_data, desc="评估进度"):
            # 预测
            response = self.predict(
                instruction=sample['instruction'],
                input_text=sample['input']
            )
            
            pred_emotion = self.extract_emotion_label(response)
            true_emotion = sample['output'].split('.')[0].strip()
            
            predictions.append(pred_emotion)
            true_labels.append(true_emotion)
        
        # 计算指标
        report = classification_report(
            true_labels, 
            predictions, 
            output_dict=True,
            zero_division=0
        )
        
        cm = confusion_matrix(true_labels, predictions, 
                             labels=["happy/excited", "stressed/anxious", 
                                     "sad/depressed", "neutral/calm"])
        
        results = {
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "predictions": predictions,
            "true_labels": true_labels
        }
        
        return results
    
    def print_results(self, results: Dict):
        """打印评估结果"""
        print("\n" + "=" * 60)
        print("评估结果")
        print("=" * 60)
        
        report = results["classification_report"]
        
        print(f"\n整体准确率: {report['accuracy']:.4f}")
        print(f"Macro F1: {report['macro avg']['f1-score']:.4f}")
        print(f"Weighted F1: {report['weighted avg']['f1-score']:.4f}")
        
        print("\n各类别详细指标:")
        print("-" * 50)
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
    parser.add_argument("--adapter_path", type=str, default="./emotion-lora-adapter")
    parser.add_argument("--test_data", type=str, default="test_split.json")
    
    args = parser.parse_args()
    
    # 加载测试数据
    with open(args.test_data, 'r') as f:
        test_data = json.load(f)
    
    # 评估
    evaluator = EmotionEvaluator(args.base_model, args.adapter_path)
    results = evaluator.evaluate_dataset(test_data)
    evaluator.print_results(results)
    
    # 保存结果
    with open("evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
```

### 6.2 推理脚本

创建 `inference.py`:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse

class EmotionInference:
    """实时情绪推断"""
    
    def __init__(self, base_model_name: str, adapter_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.to(self.device)
        self.model.eval()
    
    def analyze_hr_data(
        self, 
        hr: float, 
        rmssd: float = None, 
        sdnn: float = None, 
        pnn50: float = None
    ) -> dict:
        """分析心率数据"""
        
        # 构建输入
        input_parts = [f"Heart Rate: {hr:.1f} bpm."]
        
        if rmssd is not None:
            input_parts.append(f"Heart Rate Variability (RMSSD): {rmssd:.1f} ms.")
        if sdnn is not None:
            input_parts.append(f"SDNN: {sdnn:.1f} ms.")
        if pnn50 is not None:
            input_parts.append(f"PNN50: {pnn50:.1f}%.")
        
        input_text = " ".join(input_parts)
        
        instruction = "Based on the following physiological data, analyze the user's emotional state."
        
        prompt = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n"
            "### Input:\n{input}\n\n"
            "### Response:\n"
        ).format(instruction=instruction, input=input_text)
        
        # 生成
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                num_beams=3,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "### Response:\n" in response:
            response = response.split("### Response:\n")[-1].strip()
        
        return {
            "input": input_text,
            "analysis": response,
            "hr": hr,
            "hrv": {
                "rmssd": rmssd,
                "sdnn": sdnn,
                "pnn50": pnn50
            }
        }
    
    def interactive_mode(self):
        """交互式推理模式"""
        print("=" * 60)
        print("心率情绪分析 - 交互模式")
        print("输入 'quit' 退出")
        print("=" * 60)
        
        while True:
            try:
                hr = float(input("\n请输入心率 (bpm): "))
                
                rmssd_str = input("请输入RMSSD (ms，可选，回车跳过): ")
                rmssd = float(rmssd_str) if rmssd_str else None
                
                result = self.analyze_hr_data(hr, rmssd)
                
                print("\n" + "-" * 40)
                print("分析结果:")
                print(result['analysis'])
                print("-" * 40)
                
            except ValueError:
                print("输入格式错误，请输入数字")
            except KeyboardInterrupt:
                print("\n退出")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="MS846/fitness-gpt2-124m")
    parser.add_argument("--adapter_path", type=str, default="./emotion-lora-adapter")
    parser.add_argument("--interactive", action="store_true")
    
    # 单次推理参数
    parser.add_argument("--hr", type=float, help="心率 (bpm)")
    parser.add_argument("--rmssd", type=float, help="RMSSD (ms)")
    
    args = parser.parse_args()
    
    engine = EmotionInference(args.base_model, args.adapter_path)
    
    if args.interactive:
        engine.interactive_mode()
    elif args.hr:
        result = engine.analyze_hr_data(args.hr, args.rmssd)
        print(result['analysis'])
    else:
        print("请指定 --interactive 或 --hr 参数")
```

### 6.3 推理示例

```bash
# 交互模式
python inference.py --interactive

# 单次推理
python inference.py --hr 85 --rmssd 30

# 指定模型路径
python inference.py \
    --base_model MS846/fitness-gpt2-124m \
    --adapter_path ./emotion-lora-adapter \
    --hr 72 --rmssd 45
```

---

## 7. 故障排除与调试

### 7.1 常见问题与解决方案

#### 问题1: CUDA Out of Memory

```python
# 解决方案1: 启用4bit量化
python train.py --use_4bit ...

# 解决方案2: 减小batch_size
# 在train.py中已自动调整，或手动设置

# 解决方案3: 减小序列长度
python train.py --max_length 128 ...
```

#### 问题2: 模型不收敛

```bash
# 检查项：
# 1. 数据是否正确加载
python -c "import json; d=json.load(open('train_split.json')); print(len(d), d[0])"

# 2. 学习率是否合适
# 尝试 1e-5, 3e-5, 5e-5, 1e-4

# 3. 标签是否正确
# 确保labels中-100的位置正确
```

#### 问题3: 评估指标异常

```python
# 检查数据分布
from collections import Counter
import json

data = json.load(open('train_split.json'))
labels = [d['output'].split('.')[0] for d in data]
print(Counter(labels))

# 如果不平衡，考虑：
# 1. 使用class weights
# 2. 过采样少数类
# 3. 使用focal loss
```

### 7.2 调试工具

创建 `debug_utils.py`:

```python
import torch
import json
from typing import Dict

def check_tokenization(tokenizer, sample: Dict):
    """检查分词结果"""
    full_text = sample['instruction'] + "\n" + sample['input'] + "\n" + sample['output']
    
    encoded = tokenizer(full_text, return_tensors="pt")
    decoded = tokenizer.decode(encoded['input_ids'][0])
    
    print("原始文本长度:", len(full_text))
    print("Token数量:", encoded['input_ids'].shape[1])
    print("\n解码验证:")
    print(decoded[:500] + "...")
    
    return encoded

def visualize_training_logs(log_dir: str):
    """可视化训练日志"""
    import matplotlib.pyplot as plt
    import os
    
    log_file = os.path.join(log_dir, "trainer_state.json")
    
    if not os.path.exists(log_file):
        print(f"日志文件不存在: {log_file}")
        return
    
    with open(log_file, 'r') as f:
        state = json.load(f)
    
    logs = state['log_history']
    
    # 提取损失
    train_steps = [log['step'] for log in logs if 'loss' in log]
    train_loss = [log['loss'] for log in logs if 'loss' in log]
    
    eval_steps = [log['step'] for log in logs if 'eval_loss' in log]
    eval_loss = [log['eval_loss'] for log in logs if 'eval_loss' in log]
    
    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(train_steps, train_loss, label='Training Loss', alpha=0.7)
    if eval_steps:
        ax.plot(eval_steps, eval_loss, label='Validation Loss', marker='o')
    
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=150)
    plt.show()
```

---

## 8. 进阶优化策略

### 8.1 超参数搜索

```python
# 使用optuna进行超参数搜索
import optuna

def objective(trial):
    """Optuna目标函数"""
    
    # 搜索空间
    lora_r = trial.suggest_categorical('lora_r', [4, 8, 16, 32])
    lora_alpha = lora_r * 2
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-4, log=True)
    warmup_ratio = trial.suggest_float('warmup_ratio', 0.05, 0.2)
    max_length = trial.suggest_categorical('max_length', [128, 256, 512])
    
    # 训练...
    # 返回验证损失
    return eval_loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

print("最佳参数:", study.best_params)
```

### 8.2 模型集成

```python
class EnsemblePredictor:
    """多模型集成"""
    
    def __init__(self, adapter_paths: list):
        self.models = []
        for path in adapter_paths:
            model = self._load_model(path)
            self.models.append(model)
    
    def predict(self, input_data):
        """投票集成"""
        predictions = []
        for model in self.models:
            pred = model.predict(input_data)
            predictions.append(pred)
        
        # 多数投票
        from collections import Counter
        counter = Counter(predictions)
        return counter.most_common(1)[0][0]
```

### 8.3 持续学习

```python
# 增量训练策略
def incremental_training(
    base_adapter_path: str,
    new_data_path: str,
    output_path: str,
    epochs: int = 1
):
    """在新数据上增量训练"""
    
    # 加载已有的LoRA适配器
    model, tokenizer = load_model_with_lora(
        model_name="MS846/fitness-gpt2-124m",
        adapter_path=base_adapter_path
    )
    
    # 加载新数据
    new_dataset = EmotionDataset(new_data_path, tokenizer)
    
    # 使用较小的学习率微调
    training_args = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=epochs,
        learning_rate=1e-5,  # 较小的学习率
        ...
    )
    
    # 训练并保存
    trainer = Trainer(model=model, args=training_args, ...)
    trainer.train()
    model.save_pretrained(output_path)
```

---

## 9. 参考资源

### 9.1 学术论文
- DEAP: An EEG Database for the Analysis of Emotions (Koelstra et al., 2012)
- AMIGOS: A Dataset for Affect, Personality and Mood Research (Miranda-Correa et al., 2021)
- LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)

### 9.2 GitHub仓库
- PEFT: https://github.com/huggingface/peft
- Transformers: https://github.com/huggingface/transformers
- HeartPy: https://github.com/rhenley/heartpy

### 9.3 预训练模型
- MS846/fitness-gpt2-124m: https://huggingface.co/MS846/fitness-gpt2-124m
- roberta-base-go_emotions: https://huggingface.co/SamLowe/roberta-base-go_emotions

### 9.4 数据集下载
- DEAP: https://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html
- AMIGOS: http://www.eecs.qmul.ac.uk/mmv/datasets/amigos/download.html

---

## 附录A: 项目文件结构

```
fitness-gpt2-lora/
├── data_pipeline.py          # 数据处理
├── model_config.py           # 模型配置
├── data_collator.py          # 数据集类
├── train.py                  # 训练脚本
├── evaluate.py               # 评估脚本
├── inference.py              # 推理脚本
├── verify_env.py             # 环境验证
├── debug_utils.py            # 调试工具
├── README.md                 # 项目说明
│
├── data/                     # 原始数据
│   └── deap/
│
├── processed_data/           # 处理后的数据
│   ├── train_split.json
│   ├── val_split.json
│   └── test_split.json
│
├── emotion-lora-adapter/     # 训练好的LoRA适配器
│   ├── adapter_model.bin
│   ├── adapter_config.json
│   └── tokenizer_config.json
│
└── logs/                     # 训练日志
    └── ...
```

---

## 附录B: 快速启动命令

```bash
# 1. 环境准备
conda activate lora_emotion
pip install torch transformers peft datasets accelerate scipy numpy pandas scikit-learn

# 2. 验证环境
python verify_env.py

# 3. 处理数据 (需要先下载DEAP数据集)
python data_pipeline.py --data_dir ./data/deap --output train_data.json
python -c "from data_pipeline import split_dataset; split_dataset('train_data.json')"

# 4. 训练模型
python train.py \
    --train_data train_split.json \
    --val_data val_split.json \
    --epochs 3

# 5. 评估模型
python evaluate.py --test_data test_split.json

# 6. 使用模型
python inference.py --interactive
```

---

**文档维护**: 如有问题或建议，请更新此文档或提交Issue。
