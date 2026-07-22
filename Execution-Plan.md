# 情绪识别项目 - 执行计划 (v2)

> 版本：2.0 | 更新日期：2026-07-20
> 方案：Qwen2.5-7B-Instruct + int4 量化 + PEFT LoRA

---

## 项目概览

| 项目信息 | 详情 |
|----------|------|
| 项目名称 | 基于 Qwen2.5-7B + LoRA 的生理信号情绪识别系统 |
| 目标模型 | Qwen/Qwen2.5-7B-Instruct (7B参数, int4量化) |
| 目标平台 | Mac Mini M4 (16GB RAM, MPS) |
| 微调方法 | PEFT LoRA (r=16, alpha=32) |
| 训练框架 | HuggingFace Trainer |
| 数据规模 | 5000条合成数据（每类1250条） |

---

## 已完成阶段

### 第一阶段：环境准备 ✓
- [x] Python 3.12.13 环境
- [x] PyTorch 2.13.0 + MPS 可用
- [x] 安装依赖：transformers, peft, bitsandbytes, datasets, accelerate, scikit-learn
- [x] 创建 requirements.txt

### 第二阶段：数据生成 ✓
- [x] 设计基于生理学的严格分离规则
- [x] 各类别 RMSSD 范围无重叠：stressed(10-22), sad(24-33), neutral(35-48), happy(50-75)
- [x] 生成 5000 条高质量合成数据
- [x] 分层划分：train 3502 / val 748 / test 750
- [x] 数据质量验证通过

### 第三阶段：模型与训练脚本 ✓
- [x] train_qwen.py：int4 量化 + PEFT LoRA + SFTTrainer
- [x] ChatML 模板格式验证通过
- [x] 训练超参数配置：epochs=5, lr=2e-4, batch=2, grad_accum=8

### 第四阶段：评估与推理脚本 ✓
- [x] evaluate_qwen.py：完整评估（accuracy, precision, recall, F1, 混淆矩阵, 错误分析）
- [x] inference_qwen.py：单次/批量/交互式推理
- [x] 输入验证（SDNN >= RMSSD 生理约束）
- [x] unknown 标签处理（排除出评估指标）

---

## 当前执行阶段

### 第五阶段：模型训练   进行中
- [x] 启动训练脚本
- [ ] 下载 Qwen2.5-7B-Instruct 权重
- [ ] int4 量化加载
- [ ] LoRA 微调训练（5 epochs）
- [ ] 保存最佳 LoRA adapter

```bash
# 执行命令
python scripts/train_qwen.py
```

**训练超参数**：
| 参数 | 值 | 说明 |
|------|-----|------|
| epochs | 5 | 训练轮数 |
| learning_rate | 2e-4 | 学习率 |
| batch_size | 2 | 单卡 batch |
| gradient_accumulation | 8 | 梯度累积（有效 batch=16） |
| lr_scheduler | cosine | 余弦退火 |
| warmup_ratio | 0.1 | 预热比例 |
| lora_r | 16 | LoRA 秩 |
| lora_alpha | 32 | LoRA 缩放因子 |
| lora_dropout | 0.05 | Dropout |
| target_modules | q/k/v/o/gate/up/down_proj | 7层投影 |

---

## 待执行阶段

### 第六阶段：评估与报告
- [ ] 运行 evaluate_qwen.py 在测试集上评估
- [ ] 生成 evaluation_results_v2.json
- [ ] 生成混淆矩阵
- [ ] 错误分析
- [ ] 生成可视化报告（loss曲线、混淆矩阵热力图）

```bash
python scripts/evaluate_qwen.py
python scripts/plot_results.py
```

### 第七阶段：推理验证
- [ ] 测试用例验证
- [ ] 交互式推理测试
- [ ] 模型加载验证

```bash
# 测试用例
python scripts/inference_qwen.py
# 交互模式
python scripts/inference_qwen.py --interactive
```

### 第八阶段：交付整理
- [ ] 更新 Progress-Tracker.md
- [ ] 确认所有输出文件完整
- [ ] 模型文件清单

---

## 项目文件结构

```
fanxing/
├── Execution-Plan.md          # 本文件
├── Progress-Tracker.md        # 进度跟踪
├── requirements.txt           # Python 依赖
├── evaluation_results.json    # v1 评估结果（旧）
├── data/
│   └── processed/
│       ├── train_split.json   # 训练集 (3502条)
│       ├── val_split.json     # 验证集 (748条)
│       ├── test_split.json    # 测试集 (750条)
│       └── full_data.json     # 完整数据 (5000条)
├── models/
│   └── qwen_lora/             # LoRA adapter 输出目录
├── scripts/
│   ├── generate_data.py       # 数据生成
│   ├── data_pipeline.py       # 数据管道
│   ├── train_qwen.py          # Qwen2.5-7B 训练
│   ├── evaluate_qwen.py       # 评估脚本
│   ├── inference_qwen.py      # 推理脚本
│   ├── plot_results.py        # 可视化报告
│   └── (旧脚本保留供参考)
└── venv/                      # Python 虚拟环境
```

---

## 成功标准（v2）

| 指标 | 目标值 | 说明 |
|------|--------|------|
| Accuracy | >= 70% | 4分类（随机基线25%） |
| Macro F1 | >= 0.60 | 各类别均衡表现 |
| 各类别 F1 | >= 0.40 | 无明显偏科 |
| 训练 Loss | 收敛 | 最终 loss < 0.5 |
| 推理延迟 | < 5秒/样本 | Mac M4 int4 |

---

*计划版本: 2.0 | 下次更新: 训练完成后*
