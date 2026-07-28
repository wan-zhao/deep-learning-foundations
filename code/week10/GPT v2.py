##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~1.SimpleTokenizer~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##


class SimpleTokenizer:
	def __init__(self):
		self.special_tokens = ['[PAD]', '<bos>', '<eos>']
		self.vocab = {}
		self.inv_vocab = {}

	def build_vocab(self, texts, min_freq=1):
		# 统计所有文本中各词频次
		freq = {}
		for text in texts:
			for tok in text.lower().split():
				freq[tok] = freq.get(tok, 0) + 1
		# 保留高频词及特殊符号
		tokens = self.special_tokens + [t for t, c in freq.items() if c >= min_freq]
		self.vocab = {tok: i for i, tok in enumerate(tokens)}
		self.inv_vocab = {i: tok for tok, i in self.vocab.items()}

	def tokenize(self, text):
		return text.lower().split()

	def convert_tokens_to_ids(self, tokens):
		return [self.vocab.get(tok, self.vocab['[PAD]']) for tok in tokens]

	def convert_ids_to_tokens(self, ids):
		return [self.inv_vocab.get(i, '[PAD]') for i in ids]


##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~2.数据集构建~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##

from torch.utils.data import Dataset, DataLoader
import torch


class CLMDataset(Dataset):
	def __init__(self, texts, tokenizer, seq_len=16):
		examples = []
		pad_id = tokenizer.vocab['[PAD]']
		bos_id = tokenizer.vocab['<bos>']
		eos_id = tokenizer.vocab['<eos>']

		for txt in texts:
			ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(txt))
			# 若长度不足， PAD 对齐到 seq_len+1
			if len(ids) < seq_len + 1:
				ids += [pad_id] * (seq_len + 1 - len(ids))

			# 滑动窗口切片
			for i in range(len(ids) - seq_len):
				chunk = ids[i : i + seq_len + 1]  # 长度 seq_len+1
				# 输入前加 bos，目标末尾加 eos
				inp = [bos_id] + chunk[:-1]
				tgt = chunk[1:] + [eos_id]
				examples.append((torch.tensor(inp), torch.tensor(tgt)))

		self.examples = examples
		self.seq_len = seq_len

	def __len__(self):
		return len(self.examples)

	def __getitem__(self, i):
		return self.examples[i]


##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~3.解码器块~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##

import torch.nn as nn
import math


class DecoderBlock(nn.Module):
	def __init__(self, embed_dim, num_heads, ff_hidden, dropout=0.1):
		super().__init__()
		self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
		self.ln1 = nn.LayerNorm(embed_dim)
		self.ff = nn.Sequential(
			nn.Linear(embed_dim, ff_hidden),
			nn.GELU(),
			nn.Linear(ff_hidden, embed_dim)
		)
		self.ln2 = nn.LayerNorm(embed_dim)
		self.drop = nn.Dropout(dropout)

	def forward(self, x):
		# x: [T, B, D]
		T = x.size(0)
		# 因果掩码：屏蔽未来位置
		mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
		a, _ = self.attn(x, x, x, attn_mask=mask)  # 自注意力
		x = self.ln1(x + self.drop(a))  # 残差 + 规范化
		f = self.ff(x)  # 前馈
		return self.ln2(x + self.drop(f))  # 残差 + 规范化


##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~4.miniGPT主模型定义~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##


class MiniGPT(nn.Module):
	def __init__(self, vocab_size, seq_len=16, embed_dim=64, n_layers=2, n_heads=4,
				 ff_hidden=256):
		super().__init__()
		self.block_size = seq_len
		self.tok_emb = nn.Embedding(vocab_size, embed_dim)
		self.pos_emb = nn.Embedding(seq_len, embed_dim)
		self.layers = nn.ModuleList([
			DecoderBlock(embed_dim, n_heads, ff_hidden) for _ in range(n_layers)
		])
		self.ln_f = nn.LayerNorm(embed_dim)
		self.head = nn.Linear(embed_dim, vocab_size)

	def forward(self, x):
		# x: [B, T]
		B, T = x.size()
		assert T <= self.block_size, f"输入长度 {T} 超过 block_size {self.block_size}"
		tok = self.tok_emb(x)  # [B,T,D]
		pos = self.pos_emb(torch.arange(T, device=x.device))[None]  # [1,T,D]
		h = (tok + pos).transpose(0, 1)  # [T,B,D]
		for layer in self.layers:
			h = layer(h)
		h = self.ln_f(h.transpose(0, 1))  # [B,T,D]
		return self.head(h)  # [B,T,V]


##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~5.训练函数~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##


def train(model, dataloader, epochs=5, lr=1e-3, device='cpu'):
	opt = torch.optim.AdamW(model.parameters(), lr=lr)
	loss = nn.CrossEntropyLoss()
	model.to(device).train()
	for ep in range(1, epochs + 1):
		total, acc = 0, 0.0
		for x, y in dataloader:
			x, y = x.to(device), y.to(device)
			logits = model(x)  # [B,T,V]
			L = loss(logits.view(-1, logits.size(-1)), y.view(-1))
			opt.zero_grad()
			L.backward()
			opt.step()
			total += 1
			acc += L.item()
		print(f"Epoch {ep:2d} — loss: {acc/total:.4f}")


##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~6.文本生成~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##


import torch.nn.functional as F


@torch.no_grad()
def generate(model, tokenizer, prompt, max_len=50, strategy='greedy', **kwargs):
	"""
	自回归生成函数，支持 <bos>/<eos> 且屏蔽所有特殊 token。
	Args:
	model: 已训练好的 MiniGPT 模型
	tokenizer: SimpleTokenizer 实例，包含 special_tokens = ['[PAD]', '<bos>', '<eos>']
	prompt: str，生成的起始文本（不含 <bos>）
	max_len: int，最大生成长度（不包含 <bos>）
	strategy: 'greedy' | 'top_k' | 'top_p' | 'temperature'
	**kwargs:
		- top_k 时传 top_k=int
		- top_p 时传 top_p=float
		- temperature 时传 temperature=float (T>0，越大越随机)
	Returns:
	List[str]: 生成的 token 列表（已去掉所有特殊 token）
	"""
	device = next(model.parameters()).device
	# 特殊 token 的 id 列表
	special_ids = [tokenizer.vocab[t] for t in tokenizer.special_tokens]
	bos_id = tokenizer.vocab['<bos>']
	eos_id = tokenizer.vocab['<eos>']
	# 初始输入：在 prompt 前加上 <bos>
	prompt_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(prompt))
	x = torch.tensor([[bos_id] + prompt_ids], dtype=torch.long, device=device)  # [1, L+1]
	for _ in range(max_len):
		# 只保留最后 block_size 长度以匹配模型输入限制
		x_cond = x if x.size(1) <= model.block_size else x[:, -model.block_size:]
		# 前向，取最后一个时刻的 logits
		logits = model(x_cond)[0, -1]  # [V] model(x_cond)-->[B,T,V]
		# 屏蔽所有特殊 token（包括 [PAD], <bos>, <eos>）
		logits[special_ids] = -float('inf')

		# 根据策略选择下一个 token id
		if strategy == 'greedy':
			idx = logits.argmax()
		elif strategy == 'top_k':
			probs, idxs = F.softmax(logits, dim=-1).topk(kwargs.get('top_k', 10))
			idx = idxs[probs.multinomial(num_samples=1)]
		elif strategy == 'top_p':
			probs, idxs = F.softmax(logits, dim=-1).sort(descending=True)
			cum = probs.cumsum(0)
			mask = cum <= kwargs.get('top_p', 0.9)
			probs = probs * mask
			idx = idxs[probs.multinomial(num_samples=1)]
		elif strategy == 'temperature':
			temperature = float(kwargs.get('temperature', 1.0))
			if temperature <= 0:
				raise ValueError(f"temperature must be > 0, got {temperature}")
			probs = F.softmax(logits / temperature, dim=-1)
			idx = probs.multinomial(num_samples=1)
		else:
			raise ValueError(f"Unknown strategy: {strategy}")

		# 若生成 <eos>，停止循环
		if idx.item() == eos_id:
			break
		# 拼接到输入序列
		x = torch.cat([x, idx.unsqueeze(0)], dim=1)

	# 转回 tokens，并过滤掉所有特殊 token
	tokens = tokenizer.convert_ids_to_tokens(x[0].tolist())
	return [t for t in tokens if t not in tokenizer.special_tokens]


##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~7.示例运行~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##

# 入口脚本已拆分到 run_gpt_demo.py：
# - 便于把本文件作为模块复用（import GPT）
# - 便于后续加参数、做实验对比
