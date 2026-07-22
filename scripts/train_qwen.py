"""
Qwen2.5-7B-Instruct + int4 量化 + LoRA 情绪识别训练脚本

使用 HuggingFace Transformers + PEFT + Trainer
"""

import torch
import json
import os
import sys
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType


# Qwen ChatML 模板
SYSTEM_PROMPT = "You are an emotion classification assistant. Given physiological data, classify the user's emotional state as exactly one word: happy, stressed, sad, or neutral."

def format_chatml(sample):
    """将数据格式化为 Qwen ChatML 格式"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{sample['instruction']}\n\n{sample['input']}"},
        {"role": "assistant", "content": sample['output']},
    ]
    return messages


def preprocess_function(examples, tokenizer, max_length=512):
    """预处理数据：tokenize 并创建 labels"""
    input_ids_list = []
    labels_list = []
    attention_mask_list = []

    for i in range(len(examples['instruction'])):
        sample = {
            'instruction': examples['instruction'][i],
            'input': examples['input'][i],
            'output': examples['output'][i],
        }

        # 构建完整对话（含 assistant 回复）
        messages = format_chatml(sample)
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # 构建 prompt（不含 assistant 回复）
        prompt_messages = messages[:-1]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )

        # Tokenize
        full_enc = tokenizer(full_text, truncation=True, max_length=max_length)
        prompt_enc = tokenizer(prompt_text, truncation=True, max_length=max_length)

        input_ids = full_enc['input_ids']
        attention_mask = full_enc['attention_mask']

        # Labels: prompt 部分设为 -100（不计算 loss）
        prompt_len = len(prompt_enc['input_ids'])
        labels = [-100] * prompt_len + input_ids[prompt_len:]

        # 确保长度一致
        labels = labels[:len(input_ids)]

        input_ids_list.append(input_ids)
        labels_list.append(labels)
        attention_mask_list.append(attention_mask)

    return {
        'input_ids': input_ids_list,
        'labels': labels_list,
        'attention_mask': attention_mask_list,
    }


def load_model_and_tokenizer(model_name, use_4bit=True):
    """加载模型和分词器"""
    print(f"加载模型: {model_name}")

    is_cuda = torch.cuda.is_available()
    is_mps = torch.backends.mps.is_available()

    # int4 量化需要 CUDA
    if use_4bit and not is_cuda:
        print("  注意: 无 CUDA GPU，无法使用 int4 量化，使用 fp16")
        use_4bit = False

    # int4 量化配置（仅 CUDA 可用）
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        print("使用 int4 (NF4) 量化")
    else:
        bnb_config = None
        if is_mps:
            print("使用 fp16 模型 + MPS（TrainingArguments fp16=False）")
        else:
            print("使用 CPU fp16 训练")

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型 - 在 Mac 上显式使用 MPS 或 CPU，避免 device_map="auto" 导致的分片问题
    print("  加载模型...")
    if use_4bit:
        # CUDA int4 量化模式
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    elif is_mps:
        # Mac MPS 模式 - fp16 模型可节省一半内存
        # 注意: TrainingArguments fp16=False（GradScaler 不支持 MPS），但模型本身用 fp16 存储和计算
        print("  加载到 CPU 后移至 MPS（fp16 模型）...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="cpu",
        )
        model = model.to("mps")
        model.gradient_checkpointing_enable()
    else:
        # CPU 模式
        print("  加载到 CPU（fp16）...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="cpu",
        )
        model.gradient_checkpointing_enable()

    return model, tokenizer


def apply_lora(model, r=16, alpha=32, dropout=0.05):
    """应用 LoRA"""
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def train(args):
    """训练主函数"""
    print("=" * 60)
    print("Qwen2.5-7B-Instruct + int4 LoRA 情绪识别训练")
    print("=" * 60)

    # 加载模型
    model, tokenizer = load_model_and_tokenizer(
        args.model_name, use_4bit=args.use_4bit
    )

    # 应用 LoRA
    print("\n配置 LoRA...")
    model = apply_lora(model, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)

    # 加载数据
    print("\n加载数据集...")
    with open(args.train_data, 'r') as f:
        train_data = json.load(f)
    with open(args.val_data, 'r') as f:
        val_data = json.load(f)

    print(f"训练样本: {len(train_data)}")
    print(f"验证样本: {len(val_data)}")

    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    # 预处理
    print("\n预处理数据...")
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    val_dataset = val_dataset.map(
        lambda x: preprocess_function(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=val_dataset.column_names,
    )

    # 训练参数 - MPS 不支持 fp16 mixed precision training (需要 CUDA)
    # 在 MPS 上使用 fp32 或 bf16
    is_mps = torch.backends.mps.is_available()
    use_fp16 = not is_mps and torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        fp16=use_fp16,
        bf16=False,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=5,
        report_to="none",
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # 开始训练
    print("\n开始训练...")
    trainer.train()

    # 保存最佳模型
    print("\n保存 LoRA adapter...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\n" + "=" * 60)
    print("训练完成！")
    print(f"模型保存至: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Qwen2.5-7B LoRA 训练")

    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--train_data", type=str, default="./data/processed/train_split.json")
    parser.add_argument("--val_data", type=str, default="./data/processed/val_split.json")
    parser.add_argument("--output_dir", type=str, default="./models/qwen_lora")
    parser.add_argument("--no_4bit", action="store_true", help="禁用量化，使用 fp16")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation", type=int, default=8)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    args = parser.parse_args()
    args.use_4bit = not args.no_4bit

    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
