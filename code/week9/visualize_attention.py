import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

from transformer import (
    TransformerModel, 
    PositionalEncoding,
    MultiHeadAttention,
    PositionwiseFeedForward,
    EncoderLayer,
    Encoder,
    DecoderLayer,
    Decoder,
    generate_square_subsequent_mask,
    tokenize_en,
    tokenize_zh
)

# 修改 MultiHeadAttention 类以返回注意力权重
class MultiHeadAttentionWithWeights(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttentionWithWeights, self).__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads

        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.linear_out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        Q = self.linear_q(query)
        K = self.linear_k(key)
        V = self.linear_v(value)
        
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn_weights = attn.clone()  # 保存注意力权重
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)
        
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        out = self.linear_out(out)
        return out, attn_weights

# 加载训练好的模型权重
checkpoint = torch.load('transformer_model.pth')

word2idx_en = checkpoint['word2idx_en']
word2idx_zh = checkpoint['word2idx_zh']
idx2word_en = checkpoint['idx2word_en']
idx2word_zh = checkpoint['idx2word_zh']
max_len_en = checkpoint['max_len_en']
max_len_zh = checkpoint['max_len_zh']
model_config = checkpoint['model_config']

def encode_sentence(sentence, tokenize_fn, word2idx, max_len):
    tokens = tokenize_fn(sentence)
    token_ids = [word2idx["<BOS>"]] + [word2idx.get(token, word2idx["<UNK>"]) for token in tokens] + [word2idx["<EOS>"]]
    if len(token_ids) < max_len:
        token_ids += [word2idx["<PAD>"]] * (max_len - len(token_ids))
    else:
        token_ids = token_ids[:max_len]
    return token_ids, tokens

# 实例化模型
model = TransformerModel(
    src_vocab_size=model_config['src_vocab_size'],
    tgt_vocab_size=model_config['tgt_vocab_size'],
    d_model=model_config['d_model'],
    num_heads=model_config['num_heads'],
    num_encoder_layers=model_config['num_encoder_layers'],
    num_decoder_layers=model_config['num_decoder_layers'],
    d_ff=model_config['d_ff'],
    dropout=model_config['dropout']
)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("模型加载成功！")

# 提取注意力权重的翻译函数
def translate_with_attention(model, src_sentence, max_len=20):
    model.eval()
    src_ids, src_tokens = encode_sentence(src_sentence, tokenize_en, word2idx_en, max_len_en)
    src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0)
    
    # 获取编码器输出
    src_emb = model.src_embedding(src_tensor) * math.sqrt(model.d_model)
    src_emb = model.pos_encoder(src_emb)
    memory = model.encoder(src_emb, mask=None)
    
    tgt_ids = [word2idx_zh["<BOS>"]]
    attention_weights_list = []
    
    for i in range(max_len):
        tgt_tensor = torch.tensor(tgt_ids, dtype=torch.long).unsqueeze(0)
        tgt_mask = generate_square_subsequent_mask(tgt_tensor.size(1)).to(tgt_tensor.device).to(dtype=torch.float32)
        
        with torch.no_grad():
            tgt_emb = model.tgt_embedding(tgt_tensor) * math.sqrt(model.d_model)
            tgt_emb = model.pos_encoder(tgt_emb)
            
            # 手动通过解码器层以获取注意力权重
            x = tgt_emb
            for layer in model.decoder.layers:
                # 自注意力
                self_attn_out = layer.self_attn(x, x, x, tgt_mask)
                x = layer.norm1(x + layer.dropout1(self_attn_out))
                
                # 交叉注意力 - 这里我们需要获取注意力权重
                # 临时替换为带权重的版本
                attn_module = MultiHeadAttentionWithWeights(
                    model_config['d_model'], 
                    model_config['num_heads'], 
                    model_config['dropout']
                )
                attn_module.load_state_dict(layer.cross_attn.state_dict())
                cross_attn_out, attn_weights = attn_module(x, memory, memory, None)
                
                x = layer.norm2(x + layer.dropout2(cross_attn_out))
                ff_out = layer.feed_forward(x)
                x = layer.norm3(x + layer.dropout3(ff_out))
            
            output = model.fc_out(x)
            attention_weights_list.append(attn_weights)
        
        next_token = output[0, -1, :].argmax().item()
        tgt_ids.append(next_token)
        if next_token == word2idx_zh["<EOS>"]:
            break
    
    translated_tokens = [idx2word_zh[idx] for idx in tgt_ids if idx2word_zh[idx] not in ["<BOS>", "<EOS>"]]
    
    return " ".join(translated_tokens), attention_weights_list, src_tokens, translated_tokens

# 可视化注意力权重
def visualize_attention(attention_weights, src_tokens, tgt_tokens, layer_idx=0, head_idx=0):
    """
    可视化指定层和头的注意力权重
    
    参数:
        attention_weights: 注意力权重列表
        src_tokens: 源语言tokens
        tgt_tokens: 目标语言tokens
        layer_idx: 要可视化的层索引
        head_idx: 要可视化的注意力头索引
    """
    if len(attention_weights) == 0:
        print("没有可用的注意力权重")
        return
    
    # 获取最后一步的注意力权重
    attn = attention_weights[-1][0, head_idx].detach().cpu().numpy()
    
    # 只显示有效的tokens（去除padding）
    src_len = min(len(src_tokens) + 2, attn.shape[1])  # +2 for BOS and EOS
    tgt_len = len(tgt_tokens)
    
    attn = attn[:tgt_len, :src_len]
    
    # 创建标签
    src_labels = ['<BOS>'] + src_tokens + ['<EOS>']
    src_labels = src_labels[:src_len]
    tgt_labels = tgt_tokens[:tgt_len]
    
    # 绘制热力图
    plt.figure(figsize=(12, 8))
    sns.heatmap(attn, xticklabels=src_labels, yticklabels=tgt_labels, 
                cmap='YlOrRd', cbar=True, square=True, linewidths=0.5)
    plt.xlabel('源语言 (英文)', fontsize=12)
    plt.ylabel('目标语言 (中文)', fontsize=12)
    plt.title(f'注意力权重热力图 - 第{head_idx+1}个注意力头', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'attention_head_{head_idx}.png', dpi=300, bbox_inches='tight')
    print(f"注意力权重图已保存为 attention_head_{head_idx}.png")
    plt.show()

# 可视化所有注意力头
def visualize_all_heads(attention_weights, src_tokens, tgt_tokens, num_heads=8):
    """可视化所有注意力头"""
    if len(attention_weights) == 0:
        print("没有可用的注意力权重")
        return
    
    attn = attention_weights[-1][0].detach().cpu().numpy()
    
    src_len = min(len(src_tokens) + 2, attn.shape[2])
    tgt_len = len(tgt_tokens)
    
    src_labels = ['<BOS>'] + src_tokens + ['<EOS>']
    src_labels = src_labels[:src_len]
    tgt_labels = tgt_tokens[:tgt_len]
    
    # 创建子图
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('所有注意力头的权重分布', fontsize=16)
    
    for head_idx in range(min(num_heads, 8)):
        row = head_idx // 4
        col = head_idx % 4
        ax = axes[row, col]
        
        head_attn = attn[head_idx, :tgt_len, :src_len]
        
        sns.heatmap(head_attn, xticklabels=src_labels, yticklabels=tgt_labels,
                   cmap='YlOrRd', cbar=True, square=True, ax=ax, linewidths=0.5)
        ax.set_title(f'注意力头 {head_idx+1}')
        ax.set_xlabel('源语言')
        ax.set_ylabel('目标语言')
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('all_attention_heads.png', dpi=300, bbox_inches='tight')
    print("所有注意力头的权重图已保存为 all_attention_heads.png")
    plt.show()

# 测试翻译并可视化
test_sentence = "I am a student."
print(f"\n测试句子: {test_sentence}")
print("=" * 60)

translated, attention_weights, src_tokens, tgt_tokens = translate_with_attention(model, test_sentence)
print(f"翻译结果: {translated}")
print(f"源语言tokens: {src_tokens}")
print(f"目标语言tokens: {tgt_tokens}")

# 可视化第一个注意力头
print("\n正在生成注意力权重可视化...")
visualize_attention(attention_weights, src_tokens, tgt_tokens, head_idx=0)

# 可视化所有注意力头
visualize_all_heads(attention_weights, src_tokens, tgt_tokens, num_heads=model_config['num_heads'])

print("\n可视化完成！")
