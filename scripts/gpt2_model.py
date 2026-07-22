"""
GPT2模型定义 - 匹配fitness-gpt2-124m的权重结构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GPTModel(nn.Module):
    """匹配fitness-gpt2权重的GPT2模型"""
    
    def __init__(self, cfg):
        super().__init__()
        self.token_embedding = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_embedding = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg['drop_rate'])
        
        self.trl = nn.ModuleList([
            TransformerBlock(cfg) for _ in range(cfg['n_layers'])
        ])
        
        self.ln_f = LayerNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)
    
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.token_embedding(in_idx)
        pos_embeds = self.pos_embedding(torch.arange(seq_len, device=in_idx.device))
        x = self.drop_emb(tok_embeds + pos_embeds)
        
        for block in self.trl:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.out_head(x)
        return logits


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadAttention(
            d_in=cfg['emb_dim'],
            d_out=cfg['emb_dim'],
            context_length=cfg['context_length'],
            num_heads=cfg['n_heads'],
            dropout=cfg['drop_rate'],
            qkv_bias=cfg['qkv_bias']
        )
        self.ff = FeedForward(cfg['emb_dim'])
        self.ln1 = LayerNorm(cfg['emb_dim'])
        self.ln2 = LayerNorm(cfg['emb_dim'])
        self.drop_shortcut = nn.Dropout(cfg['drop_rate'])
    
    def forward(self, x):
        shortcut = x
        x = self.ln1(x)
        x = self.attn(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        
        shortcut = x
        x = self.ln2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, num_heads, dropout=0.0, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.linear_out = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.tril(torch.ones(context_length, context_length)))
    
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        queries = self.W_q(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.W_k(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.W_v(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens] == 0, -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context_vec = (attn_weights @ values).transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        return self.linear_out(context_vec)


class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
    
    def forward(self, x):
        return self.layers(x)


class LayerNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = 1e-5
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * norm_x + self.beta


def generate(model, idx, max_new_tokens, context_size, eos_id=50256, top_k=None, temperature=1.0):
    """自回归生成"""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        
        logits = logits[:, -1, :] / temperature
        
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')
        
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        if idx_next.item() == eos_id:
            break
        
        idx = torch.cat((idx, idx_next), dim=1)
    
    return idx


if __name__ == "__main__":
    import tiktoken
    from huggingface_hub import hf_hub_download
    
    BASE_CONFIG = {
        'vocab_size': 50257, 'context_length': 1024, 'drop_rate': 0.0,
        'qkv_bias': True, 'emb_dim': 768, 'n_layers': 12, 'n_heads': 12,
    }
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'设备: {device}')
    
    ckpt_path = hf_hub_download(repo_id='MS846/fitness-gpt2-124m', filename='model_fitness_small.pth')
    print(f'模型路径: {ckpt_path}')
    
    model = GPTModel(BASE_CONFIG)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device).eval()
    print(f'模型加载成功！参数量: {sum(p.numel() for p in model.parameters()):,}')
    
    tokenizer = tiktoken.get_encoding('gpt2')
    print(f'词表大小: {tokenizer.n_vocab}')
    
    # 测试推理
    prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Based on the following physiological data, analyze the user's emotional state.

### Input:
Heart Rate: 85 bpm. Heart Rate Variability (RMSSD): 30 ms.

### Response:
"""
    
    ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
    out_ids = generate(model=model, idx=ids, max_new_tokens=200, context_size=1024, eos_id=50256, top_k=50, temperature=0.7)
    response = tokenizer.decode(out_ids[0].tolist())[len(prompt):]
    print(f'\n测试响应:\n{response}')
