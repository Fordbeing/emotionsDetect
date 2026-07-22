# emotionsDetect

心率情绪识别系统，基于 **Qwen2.5-1.5B-Instruct + LoRA** 实现，通过心率（HR）和心率变异性（HRV）数据识别用户情绪状态。

本项目提供：

- 一个可直接启动的 **FastAPI 推理服务**
- 已训练好的 **LoRA adapter**
- 自动下载并缓存 **基础模型** 的机制
- 单条推理、批量推理、健康检查接口
- Windows / macOS / Linux 启动说明

---

## 项目特点

- **无需重新训练**：直接使用已训练好的 LoRA 参数
- **自动获取基础模型**：本地没有 `models/qwen_base` 时会自动下载
- **离线优先**：基础模型存在时可离线启动
- **接口清晰**：提供 `/health`、`/predict`、`/predict/batch`
- **适合部署**：启动后可直接接入前端或其他服务

---

## 模型说明

本项目采用分层加载方式：

1. **基础模型**：`Qwen/Qwen2.5-1.5B-Instruct`
2. **LoRA 参数**：`models/qwen_lora/`

运行时会先加载基础模型，再叠加 LoRA adapter。

### 本地目录约定

```text
models/
├── qwen_base/     # 基础模型（可自动下载，也可手动放置）
└── qwen_lora/     # 已训练好的 LoRA adapter
```

> 说明：仓库中保留 LoRA 参数；基础模型可以在首次启动时自动下载到本地缓存目录。

---

## 环境要求

- Python 3.10+
- Git
- Windows / macOS / Linux
- 建议内存 16GB+
- GPU 可选；CPU 也可运行，但推理会更慢

---

## 快速开始

### 1. 克隆项目

```bash
git clone git@github.com:Fordbeing/emotionsDetect.git
cd emotionsDetect
```

### 2. 创建虚拟环境

Windows：

```powershell
python -m venv venv
venv\Scripts\activate
```

macOS / Linux：

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 启动服务

Windows：

```powershell
$env:HF_HUB_OFFLINE=1
$env:TRANSFORMERS_OFFLINE=1
.\venv\Scripts\python.exe api.py
```

macOS / Linux：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 venv/bin/python api.py
```

首次启动时，如果本地没有基础模型，程序会自动下载到 `models/qwen_base/`。

### 5. 打开接口文档

浏览器访问：

```text
http://127.0.0.1:8000/docs
```

健康检查：

```text
http://127.0.0.1:8000/health
```

---

## API 说明

### `GET /health`

返回服务状态、模型加载状态和运行设备信息。

示例响应：

```json
{
  "status": "ok",
  "model_loaded": true,
  "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
  "adapter_path": "models/qwen_lora",
  "device": "cpu"
}
```

### `POST /predict`

单条情绪识别。

请求体示例：

```json
{
  "hr": 75,
  "rmssd": 40,
  "sdnn": 50,
  "pnn50": 20,
  "extra_info": {
    "user_id": "u123",
    "source": "apple_watch"
  }
}
```

响应字段包含：

- `emotion`：识别结果
- `emotion_detail`：中文标签、描述、颜色、参考范围、建议
- `raw_model_output`：模型原始输出
- `confidence_note`：置信提示

### `POST /predict/batch`

批量识别，最多 50 条样本。

请求体示例：

```json
{
  "samples": [
    {"hr": 55, "rmssd": 25},
    {"hr": 78, "rmssd": 60, "pnn50": 38}
  ]
}
```

---

## 情绪标签定义

当前支持 4 类标签：

- `happy`：开心
- `stressed`：压力/紧张
- `sad`：悲伤/低落
- `neutral`：平静/中性

模型返回英文标签，接口同时给出中文说明和建议。

---

## 项目结构

```text
├── api.py                  # FastAPI 推理服务入口
├── STARTUP.md              # 完整启动指南
├── requirements.txt        # Python 依赖
├── models/
│   ├── qwen_lora/          # LoRA adapter
│   └── qwen_base/          # 基础模型（自动下载）
└── scripts/
    ├── inference_qwen.py   # 命令行推理
    ├── train_qwen.py       # LoRA 训练脚本
    ├── evaluate_qwen.py    # 模型评估
    └── verify_model.py     # 模型验证
```

---

## 常用脚本

### 交互式推理

Windows：

```powershell
.\venv\Scripts\python.exe scripts\inference_qwen.py --interactive
```

macOS / Linux：

```bash
venv/bin/python scripts/inference_qwen.py --interactive
```

### 单次推理

Windows：

```powershell
.\venv\Scripts\python.exe scripts\inference_qwen.py --hr 95 --rmssd 18
```

macOS / Linux：

```bash
venv/bin/python scripts/inference_qwen.py --hr 95 --rmssd 18
```

### 验证模型文件

```powershell
.\venv\Scripts\python.exe scripts\verify_model.py --adapter_path .\models\qwen_lora --base_model .\models\qwen_base
```

---

## 部署提示

- 如果端口 `8000` 被占用，可以修改 `api.py` 最后一行的端口号
- 如果模型下载速度较慢，可以配置 Hugging Face 镜像或登录 Token
- 如果本地已经有 `models/qwen_base`，服务会优先使用本地文件

---

## 常见问题

### 1. 启动时报找不到基础模型

请确认：

- `models/qwen_lora/adapter_config.json` 存在
- `models/qwen_lora/adapter_model.safetensors` 存在
- 首次启动时允许程序下载 `Qwen/Qwen2.5-1.5B-Instruct`

### 2. 端口占用

把 `api.py` 中的端口改成其他值，例如：

```python
uvicorn.run(app, host="0.0.0.0", port=8001)
```

### 3. Python 命令不可用

确保 Python 已安装并加入 PATH；Windows 还需要关闭应用执行别名里的 `python.exe` / `python3.exe` 冲突项。

---

## 许可与说明

本项目使用 Hugging Face 上的 Qwen 基座模型与本地训练得到的 LoRA 参数。请遵守模型与数据集对应的许可协议。
