# 心率情绪识别项目 - 启动指南

## 项目说明

本项目使用 **Qwen2.5-1.5B-Instruct** 作为基础模型，并加载已经训练好的 **LoRA adapter** 做心率情绪识别。

启动时需要两部分：

1. **基础模型**：`models/qwen_base/`
2. **LoRA 参数**：`models/qwen_lora/`

如果只有 LoRA，没有基础模型，服务无法启动。

---

## 环境要求

- Python 3.10+
- Git
- Windows / macOS / Linux
- 建议内存 16GB+
- GPU 可选，CPU 也能运行，但会更慢

---

## 推荐启动方式

### 1. 准备项目

如果你还没有把模型文件放好，请确认目录如下：

```text
D:\program\emotionsDetect\models\qwen_lora
D:\program\emotionsDetect\models\qwen_base
```

其中：

- `models/qwen_lora`：LoRA adapter
- `models/qwen_base`：Qwen2.5-1.5B-Instruct 基座模型

> 当前 `api.py` 会优先从本地加载这两个目录，不再强制联网下载。

---

## Windows 启动步骤

### 1. 打开 PowerShell

进入项目根目录：

```powershell
cd D:\program\emotionsDetect
```

### 2. 创建虚拟环境

```powershell
python -m venv venv
```

### 3. 激活虚拟环境

```powershell
venv\Scripts\activate
```

### 4. 安装依赖

```powershell
pip install -r requirements.txt
```

### 5. 如果还没有基础模型，先下载到本地

如果 `models/qwen_base/` 还不存在，先执行下面的下载命令：

```powershell
@'
from huggingface_hub import snapshot_download
from pathlib import Path

repo_id = "Qwen/Qwen2.5-1.5B-Instruct"
target_dir = Path(r"D:\program\emotionsDetect\models\qwen_base")
target_dir.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id=repo_id,
    local_dir=str(target_dir),
    local_dir_use_symlinks=False,
)
print("Download complete.")
'@ | .\venv\Scripts\python.exe -
```

### 6. 启动 API

```powershell
$env:HF_HUB_OFFLINE=1
$env:TRANSFORMERS_OFFLINE=1
.\venv\Scripts\python.exe api.py
```

也可以写成一行：

```powershell
$env:HF_HUB_OFFLINE=1; $env:TRANSFORMERS_OFFLINE=1; .\venv\Scripts\python.exe api.py
```

### 7. 验证服务

浏览器打开：

```text
http://127.0.0.1:8000/docs
```

健康检查：

```powershell
Invoke-WebRequest http://127.0.0.1:8000/health -UseBasicParsing
```

---

## macOS / Linux 启动步骤

### 1. 进入项目目录

```bash
cd /path/to/emotionsDetect
```

### 2. 创建虚拟环境

```bash
python3 -m venv venv
```

### 3. 激活虚拟环境

```bash
source venv/bin/activate
```

### 4. 安装依赖

```bash
pip install -r requirements.txt
```

### 5. 下载基础模型（如果本地还没有）

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
from pathlib import Path

repo_id = "Qwen/Qwen2.5-1.5B-Instruct"
target_dir = Path("./models/qwen_base")
target_dir.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id=repo_id,
    local_dir=str(target_dir),
    local_dir_use_symlinks=False,
)
print("Download complete.")
PY
```

### 6. 启动 API

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 venv/bin/python api.py
```

### 7. 验证服务

```bash
curl http://127.0.0.1:8000/health
```

浏览器打开：

```text
http://127.0.0.1:8000/docs
```

---

## API 接口

### `GET /health`

返回模型加载状态。

### `POST /predict`

单条推理。

请求示例：

```json
{
  "hr": 75,
  "rmssd": 40,
  "sdnn": 50,
  "pnn50": 20,
  "extra_info": {
    "user_id": "u123"
  }
}
```

### `POST /predict/batch`

批量推理，最多 50 条。

请求示例：

```json
{
  "samples": [
    {"hr": 55, "rmssd": 25},
    {"hr": 78, "rmssd": 60, "pnn50": 38}
  ]
}
```

---

## 其他脚本

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

## 常见问题

### 1. `python` 命令不可用

如果 `python --version` 失败，检查：

- Python 是否已安装
- 是否把 Python 加入 PATH
- 是否关闭了 Windows 的 `python.exe` / `python3.exe` 应用执行别名

### 2. 启动时报找不到基础模型

确认下面目录存在：

- `models/qwen_base/config.json`
- `models/qwen_base/model.safetensors`
- `models/qwen_lora/adapter_config.json`
- `models/qwen_lora/adapter_model.safetensors`

如果基础模型不在本地，API 会启动失败。

### 3. 端口 8000 被占用

修改 `api.py` 最后一行：

```python
uvicorn.run(app, host="0.0.0.0", port=8001)
```

然后重新启动。

### 4. 下载模型太慢

可以使用 Hugging Face 镜像或登录 Token 提高速度。

例如设置镜像：

```powershell
$env:HF_ENDPOINT="https://hf-mirror.com"
```

或者先登录：

```powershell
huggingface-cli login
```

### 5. `/health` 返回 500

如果 API 已经能启动，但 `/health` 仍报错，说明代码里还有运行时异常。此时请查看启动日志，确认模型是否完整加载。

---

## 一键启动顺序总结

Windows 下最常用的是这组命令：

```powershell
cd D:\program\emotionsDetect
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
$env:HF_HUB_OFFLINE=1
$env:TRANSFORMERS_OFFLINE=1
venv\Scripts\python.exe api.py
```

启动后访问：

```text
http://127.0.0.1:8000/docs
```

---

## 目录说明

```text
api.py                    # FastAPI 服务入口
models/qwen_base/         # 基础模型（本地）
models/qwen_lora/         # LoRA adapter
scripts/inference_qwen.py # 交互/单次推理
scripts/verify_model.py   # 模型加载验证
requirements.txt          # 依赖
```
