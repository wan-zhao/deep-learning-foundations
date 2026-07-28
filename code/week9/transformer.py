import math
import re
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~数据处理~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1.1 从JSON文件加载平行翻译数据
def load_training_data(json_file='training_data.json'):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    parallel_data = [(item['en'], item['zh']) for item in data['parallel_data']]
    return parallel_data

parallel_data = load_training_data()

# 1.2 定义分词函数
def tokenize_en(sentence):
    sentence = sentence.lower()
    tokens = re.findall(r"[a-zA-Z]+|[,.!?']", sentence)
    return tokens
def tokenize_zh(sentence):
    # 这里假设中文句子已用空格分词
    return sentence.split()

# 1.3 构造特殊标记及词表
SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
# 英文词表
all_tokens_en = []
for en, _ in parallel_data:
    all_tokens_en.extend(tokenize_en(en))
counter_en = Counter(all_tokens_en)
sorted_tokens_en = sorted(counter_en, key=counter_en.get, reverse=True)
vocab_en = SPECIAL_TOKENS + sorted_tokens_en
word2idx_en = {w: i for i, w in enumerate(vocab_en)}
idx2word_en = {i: w for w, i in word2idx_en.items()}
# 中文词表
all_tokens_zh = []
for _, zh in parallel_data:
    all_tokens_zh.extend(tokenize_zh(zh))
counter_zh = Counter(all_tokens_zh)
sorted_tokens_zh = sorted(counter_zh, key=counter_zh.get, reverse=True)
vocab_zh = SPECIAL_TOKENS + sorted_tokens_zh
word2idx_zh = {w: i for i, w in enumerate(vocab_zh)}
idx2word_zh = {i: w for w, i in word2idx_zh.items()}

# 1.4 设定最大长度（包含<BOS>与<EOS>）并编码句子
max_len_en = max(len(tokenize_en(en)) for en, _ in parallel_data) + 2
max_len_zh = max(len(tokenize_zh(zh)) for _, zh in parallel_data) + 2
def encode_sentence(sentence, tokenize_fn, word2idx, max_len):
    tokens = tokenize_fn(sentence)
    token_ids = [word2idx["<BOS>"]] + [word2idx.get(token, word2idx["<UNK>"]) for token in tokens] + [word2idx["<EOS>"]]
    if len(token_ids) < max_len:
        token_ids += [word2idx["<PAD>"]] * (max_len - len(token_ids))
    else:
        token_ids = token_ids[:max_len]
    return token_ids

encoded_data = []
for en, zh in parallel_data:
    en_ids = encode_sentence(en, tokenize_en, word2idx_en, max_len_en)
    zh_ids = encode_sentence(zh, tokenize_zh, word2idx_zh, max_len_zh)
    encoded_data.append((en_ids, zh_ids))

encoded_en = torch.tensor([pair[0] for pair in encoded_data], dtype=torch.long)
encoded_zh = torch.tensor([pair[1] for pair in encoded_data], dtype=torch.long)

#print("Encoded English sentences:\n", encoded_en)
#print("Encoded Chinese sentences:\n", encoded_zh)
#~~~~~~~~~~~~~~~~~~~~2.位置编码~~~~~~~~~~~~~~~~~~~~~
# 2.1 位置编码（与论文中的公式一致）
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model) # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float() # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) /d_model))
        pe[:, 0::2] = torch.sin(position * div_term) # 偶数维使用 sin
        pe[:, 1::2] = torch.cos(position * div_term) # 奇数维使用 cos
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    
#~~~~~~~~~~~~~~~~~~~~~~~3.多头注意力机制~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads

        # 对 Q, K, V 分别进行线性变换
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.linear_out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        # 线性映射
        Q = self.linear_q(query)
        K = self.linear_k(key)
        V = self.linear_v(value)
        # 分多头 reshape： (batch, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # 计算缩放点积注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            # mask: (batch, num_heads, seq_len, seq_len)或能够广播到该形状
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1) #形状(B, H, L, L)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)#形状 (B, H, L, d_k)
        # 合并各个头， reshape 回 (batch, seq_len, d_model)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model) #形状 (B, L, d_model)
        out = self.linear_out(out)
        return out
#~~~~~~~~~~~~~~~~~~~~~~~4.前馈神经网络~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 4 位置前馈网络
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))
#~~~~~~~~~~~~~~~~~~~~~~~5.Encoder layer ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 5 Encoder 层（包含自注意力和前馈网络，并有残差连接和 LayerNorm）
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_out))
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_out))
        return x
#~~~~~~~~~~~~~~~~~~~~~~~6.Encoder（堆叠多个Encoder Layer)~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 6 Encoder（堆叠多个 EncoderLayer）
class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(
        [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x
#~~~~~~~~~~~~~~~~~~~~~~~7.Decoder Layer~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 7 Decoder 层（包含 masked 自注意力、交叉注意力和前馈网络）
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        self_attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_out))
        cross_attn_out = self.cross_attn(x, memory, memory, memory_mask)
        x = self.norm2(x + self.dropout2(cross_attn_out))
        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_out))
        return x
#~~~~~~~~~~~~~~~~~~~~~~~8.Decoder（堆叠多个Decoder Layer)~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 8 Decoder（堆叠多个 DecoderLayer）
class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        return x
#~~~~~~~~~~~~~~~~~~~~~~~9.完整的Transformer模型~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 9 整体 Transformer 模型
class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads,num_encoder_layers, num_decoder_layers, d_ff, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.encoder = Encoder(num_encoder_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(num_decoder_layers, d_model, num_heads, d_ff, dropout)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        memory = self.encoder(src_emb, mask=src_mask)
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        logits = self.fc_out(output)
        return logits
    
    # 补充：构造下三角 mask，用于 Decoder 自注意力
def generate_square_subsequent_mask(sz):
    # 生成 (sz, sz) 下三角矩阵， True 表示允许， False 表示屏蔽
    mask = torch.tril(torch.ones(sz, sz)).bool()
    return mask
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~10、定义损失函数~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 10 使用交叉熵损失，忽略 <PAD> 的误差
# 注意：目标张量形状应为 (batch * seq_len,)； logits 需 reshape 为 (batch * seq_len, vocab_size)
criterion = nn.CrossEntropyLoss(ignore_index=word2idx_zh["<PAD>"])
# 3.2 简化版学习率调度器（warmup 后固定）
class SimpleLRScheduler:
    def __init__(self, optimizer, warmup_steps, base_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.step_num = 0

    def step(self):
        self.step_num += 1
        if self.step_num < self.warmup_steps:
            lr = self.base_lr * self.step_num / self.warmup_steps
        else:
            lr = self.base_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~11、训练过程~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    # 超参数设置（优化后）
    d_model = 128  # 增加模型维度
    num_heads = 8  # 增加注意力头数
    num_encoder_layers = 3  # 增加编码器层数
    num_decoder_layers = 3  # 增加解码器层数
    d_ff = 512  # 增加前馈网络维度
    dropout = 0.1
    src_vocab_size = len(vocab_en)
    tgt_vocab_size = len(vocab_zh)
    # 实例化模型
    model = TransformerModel(src_vocab_size, tgt_vocab_size, d_model, num_heads,num_encoder_layers, num_decoder_layers, d_ff, dropout)
    # 使用 Adam 优化器，调整学习率
    optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = SimpleLRScheduler(optimizer, warmup_steps=1000, base_lr=5e-4)  # 降低学习率，减少warmup步数
    # 开始训练（增加训练轮数）
    epochs = 300

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        # 此处为了演示，简单地每个样本单独更新
        for src_sentence, tgt_sentence in zip(encoded_en, encoded_zh):
            # 构造 batch
            src_input = src_sentence.unsqueeze(0) # (1, seq_len_src)
            # 对目标：教师强制， Decoder 输入为去掉最后一个 token
            tgt_input = tgt_sentence[:-1].unsqueeze(0) # (1, seq_len_tgt-1)
            tgt_output = tgt_sentence[1:].unsqueeze(0) # (1, seq_len_tgt-1)
            tgt_seq_len = tgt_input.size(1)
            # 生成下三角 mask，并转换为 float（1.0 表示允许， 0.0 表示屏蔽）
            tgt_mask =generate_square_subsequent_mask(tgt_seq_len).to(tgt_input.device).to(dtype=torch.float32)
            
            # 前向传播
            logits = model(src_input, tgt_input, src_mask=None, tgt_mask=tgt_mask)
            # logits: (1, seq_len_tgt-1, tgt_vocab_size)
            logits = logits.reshape(-1, tgt_vocab_size)
            tgt_output = tgt_output.reshape(-1)

            loss = criterion(logits, tgt_output)
            scheduler.zero_grad()
            loss.backward()
            scheduler.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(encoded_en):.4f}")

    # 保存训练好的模型权重
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'vocab_en': vocab_en,
        'vocab_zh': vocab_zh,
        'word2idx_en': word2idx_en,
        'word2idx_zh': word2idx_zh,
        'idx2word_en': idx2word_en,
        'idx2word_zh': idx2word_zh,
        'max_len_en': max_len_en,
        'max_len_zh': max_len_zh,
        'model_config': {
            'd_model': d_model,
            'num_heads': num_heads,
            'num_encoder_layers': num_encoder_layers,
            'num_decoder_layers': num_decoder_layers,
            'd_ff': d_ff,
            'dropout': dropout,
            'src_vocab_size': src_vocab_size,
            'tgt_vocab_size': tgt_vocab_size
        }
    }, 'transformer_model.pth')
    print("模型权重已保存到 transformer_model.pth")
