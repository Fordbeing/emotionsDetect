"""
LoRA实现 - 为fitness-gpt2添加低秩适配
"""

import torch
import torch.nn as nn
import math


class LoRALinear(nn.Module):
    """LoRA适配的线性层"""
    
    def __init__(self, original_linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.original = original_linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        
        # 冻结原始权重
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False
        
        # LoRA矩阵
        self.lora_A = nn.Parameter(torch.randn(r, in_features) / math.sqrt(r))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        # 原始输出
        original_output = self.original(x)
        
        # LoRA输出
        lora_output = (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        
        return original_output + lora_output


def apply_lora(model, r=8, alpha=16, dropout=0.0, target_modules=None):
    """
    将LoRA应用到模型的指定模块
    
    Args:
        model: GPT2模型
        r: LoRA秩
        alpha: 缩放因子
        dropout: Dropout率
        target_modules: 要应用LoRA的模块名称列表
    """
    if target_modules is None:
        target_modules = ['W_q', 'W_k', 'W_v']
    
    lora_params = 0
    frozen_params = 0
    
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # 替换为LoRA适配的线性层
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = dict(model.named_modules())[parent_name]
                
                lora_layer = LoRALinear(module, r=r, alpha=alpha, dropout=dropout)
                # 移动到与原模块相同的设备
                lora_layer = lora_layer.to(module.weight.device)
                setattr(parent, child_name, lora_layer)
                
                lora_params += lora_layer.lora_A.numel() + lora_layer.lora_B.numel()
                print(f'  LoRA applied to: {name}')
    
    # 冻结非LoRA参数
    for name, param in model.named_parameters():
        if 'lora_' not in name:
            param.requires_grad = False
            frozen_params += param.numel()
        else:
            param.requires_grad = True
    
    print(f'\nLoRA参数: {lora_params:,}')
    print(f'冻结参数: {frozen_params:,}')
    print(f'可训练比例: {100 * lora_params / (lora_params + frozen_params):.2f}%')
    
    return model


def save_lora(model, path):
    """保存LoRA权重"""
    lora_state = {}
    for name, param in model.named_parameters():
        if 'lora_' in name:
            lora_state[name] = param.data
    torch.save(lora_state, path)
    print(f'LoRA权重已保存到: {path}')


def load_lora(model, path):
    """加载LoRA权重"""
    lora_state = torch.load(path, map_location='cpu')
    model_dict = model.state_dict()
    
    # 只加载LoRA参数
    for name, param in lora_state.items():
        if name in model_dict:
            model_dict[name] = param
    
    model.load_state_dict(model_dict)
    print(f'LoRA权重已加载: {path}')
    return model


if __name__ == "__main__":
    from huggingface_hub import hf_hub_download
    from gpt2_model import GPTModel
    
    BASE_CONFIG = {
        'vocab_size': 50257, 'context_length': 1024, 'drop_rate': 0.0,
        'qkv_bias': True, 'emb_dim': 768, 'n_layers': 12, 'n_heads': 12,
    }
    
    device = 'cpu'
    ckpt_path = hf_hub_download(repo_id='MS846/fitness-gpt2-124m', filename='model_fitness_small.pth')
    
    model = GPTModel(BASE_CONFIG)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    
    print('应用LoRA...')
    model = apply_lora(model, r=8, alpha=16, dropout=0.05, target_modules=['W_q', 'W_k', 'W_v'])
    
    print(f'\n总参数: {sum(p.numel() for p in model.parameters()):,}')
    print(f'可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
