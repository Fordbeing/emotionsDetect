"""
推理脚本
使用训练好的LoRA模型进行情绪识别推理
"""

import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class EmotionInference:
    """情绪推断引擎"""
    
    def __init__(self, base_model_name: str, adapter_path: str):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"模型加载完成，设备: {self.device}")
    
    def analyze_hr_data(
        self, 
        hr: float, 
        rmssd: float = None, 
        sdnn: float = None, 
        pnn50: float = None
    ) -> dict:
        """分析心率数据"""
        
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
            "hrv": {"rmssd": rmssd, "sdnn": sdnn, "pnn50": pnn50}
        }
    
    def interactive_mode(self):
        """交互式模式"""
        print("="*60)
        print("心率情绪分析 - 交互模式")
        print("输入 'quit' 退出")
        print("="*60)
        
        while True:
            try:
                hr = float(input("\n请输入心率 (bpm): "))
                
                rmssd_str = input("请输入RMSSD (ms，可选，回车跳过): ")
                rmssd = float(rmssd_str) if rmssd_str else None
                
                result = self.analyze_hr_data(hr, rmssd)
                
                print("\n" + "-"*40)
                print("分析结果:")
                print(result['analysis'])
                print("-"*40)
                
            except ValueError:
                print("输入格式错误，请输入数字")
            except KeyboardInterrupt:
                print("\n退出")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="MS846/fitness-gpt2-124m")
    parser.add_argument("--adapter_path", type=str, default="./models/adapter")
    parser.add_argument("--interactive", action="store_true")
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
        print("用法: python inference.py --interactive 或 python inference.py --hr 85 --rmssd 30")
