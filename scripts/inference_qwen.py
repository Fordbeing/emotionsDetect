"""
Qwen2.5-7B-Instruct 推理脚本

支持单次推理、批量推理、交互式模式。
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


SYSTEM_PROMPT = "You are an emotion classification assistant. Given physiological data, classify the user's emotional state as exactly one word: happy, stressed, sad, or neutral."
INSTRUCTION = "Based on the following physiological data, classify the user's emotional state as one of: happy, stressed, sad, neutral."

VALID_LABELS = {"happy", "stressed", "sad", "neutral"}


def extract_label(response: str) -> str:
    """从输出提取标签"""
    r = response.lower().strip()
    if r in VALID_LABELS:
        return r
    for label in VALID_LABELS:
        if label in r:
            return label
    return "unknown"


class EmotionClassifier:
    """情绪分类器"""

    def __init__(self, base_model="Qwen/Qwen2.5-1.5B-Instruct", adapter_path="./models/qwen_lora", use_4bit=True):
        print(f"加载模型: {base_model}")

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

        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model, trust_remote_code=True, padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        is_mps = torch.backends.mps.is_available()

        if is_mps:
            print("  加载到 CPU 后移至 MPS...")
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="cpu",
            )
            model = model.to("mps")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )

        print(f"加载 adapter: {adapter_path}")
        self.model = PeftModel.from_pretrained(model, adapter_path)
        self.model.eval()
        print("模型加载完成")

    def predict(self, hr: float, rmssd: float = None, sdnn: float = None, pnn50: float = None) -> dict:
        """单条推理"""
        # 输入验证
        if sdnn is not None and rmssd is not None and sdnn < rmssd:
            print(f"  警告: SDNN ({sdnn}) < RMSSD ({rmssd})，不符合生理约束，已自动调整")
            sdnn = rmssd + 10

        # 构建输入
        parts = [f"Heart Rate: {hr:.1f} bpm."]
        if rmssd is not None:
            parts.append(f"RMSSD: {rmssd:.1f} ms.")
        if sdnn is not None:
            parts.append(f"SDNN: {sdnn:.1f} ms.")
        if pnn50 is not None:
            parts.append(f"PNN50: {pnn50:.1f}%.")
        input_text = " ".join(parts)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{INSTRUCTION}\n\n{input_text}"},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        raw_response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        label = extract_label(raw_response)

        return {
            "input": input_text,
            "emotion": label,
            "raw_output": raw_response,
        }

    def predict_batch(self, samples: list) -> list:
        """批量推理"""
        results = []
        for sample in samples:
            result = self.predict(
                hr=sample.get('hr', 70),
                rmssd=sample.get('rmssd'),
                sdnn=sample.get('sdnn'),
                pnn50=sample.get('pnn50'),
            )
            results.append(result)
        return results

    def interactive(self):
        """交互式模式"""
        print("=" * 60)
        print("心率情绪识别 - 交互模式")
        print("输入 'quit' 退出")
        print("=" * 60)

        while True:
            try:
                hr_str = input("\n心率 (bpm): ").strip()
                if hr_str.lower() == 'quit':
                    break
                hr = float(hr_str)

                rmssd_str = input("RMSSD (ms, 回车跳过): ").strip()
                rmssd = float(rmssd_str) if rmssd_str else None

                sdnn_str = input("SDNN (ms, 回车跳过): ").strip()
                sdnn = float(sdnn_str) if sdnn_str else None

                pnn50_str = input("PNN50 (%, 回车跳过): ").strip()
                pnn50 = float(pnn50_str) if pnn50_str else None

                result = self.predict(hr, rmssd, sdnn, pnn50)

                print(f"\n  情绪: {result['emotion']}")
                print(f"  模型输出: {result['raw_output']}")

            except ValueError:
                print("输入格式错误，请输入数字")
            except KeyboardInterrupt:
                print("\n退出")
                break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Qwen2.5-7B 情绪识别推理")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter_path", type=str, default="./models/qwen_lora")
    parser.add_argument("--no_4bit", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--hr", type=float, help="心率 (bpm)")
    parser.add_argument("--rmssd", type=float, help="RMSSD (ms)")
    parser.add_argument("--sdnn", type=float, help="SDNN (ms)")
    parser.add_argument("--pnn50", type=float, help="PNN50 (%)")
    args = parser.parse_args()
    use_4bit = not args.no_4bit

    classifier = EmotionClassifier(args.base_model, args.adapter_path, use_4bit)

    if args.interactive:
        classifier.interactive()
    elif args.hr is not None:
        result = classifier.predict(args.hr, args.rmssd, args.sdnn, args.pnn50)
        print(f"情绪: {result['emotion']}")
        print(f"输入: {result['input']}")
    else:
        # 默认测试用例
        test_cases = [
            {"hr": 95, "rmssd": 18, "sdnn": 22, "pnn50": 8, "expected": "stressed"},
            {"hr": 55, "rmssd": 25, "sdnn": 30, "pnn50": 10, "expected": "sad"},
            {"hr": 70, "rmssd": 42, "sdnn": 52, "pnn50": 25, "expected": "neutral"},
            {"hr": 78, "rmssd": 60, "sdnn": 70, "pnn50": 38, "expected": "happy"},
        ]
        print("测试用例:")
        print("-" * 50)
        for tc in test_cases:
            result = classifier.predict(tc['hr'], tc['rmssd'], tc['sdnn'], tc['pnn50'])
            status = "OK" if result['emotion'] == tc['expected'] else "FAIL"
            print(f"  [{status}] HR={tc['hr']}, RMSSD={tc['rmssd']} -> {result['emotion']} (expected: {tc['expected']})")
