"""Run script for GPT.py demo.

This file keeps GPT.py as a reusable module (model/dataset/tokenizer),
while this script is the executable entry point.
"""

import torch
from torch.utils.data import DataLoader

from GPT import SimpleTokenizer, CLMDataset, MiniGPT, train, generate


def main():
	# 示例文本
	texts = [
		"the quick brown fox jumps over the lazy dog",
		"hello how are you doing today",
		"this is a simple example sentence",
		"machine learning is changing the world",
		"what time is the meeting tomorrow",
		"please send me the report by friday",
		"the weather is really nice today",
		"i enjoy reading books on weekends",
		"let's grab coffee sometime next week",
		"python is my favorite programming language",
		"have you seen the latest movie",
		"my phone battery is almost dead",
		"the project deadline is approaching fast",
		"i need to buy groceries after work",
		"learning new skills is always beneficial",
		"can you help me with this problem",
		"the train was delayed by thirty minutes",
		"i love listening to music while working",
		"what are your plans for the weekend",
		"this restaurant has amazing food",
	]

	# 配置：模型 block_size 与数据集 seq_len 的关系
	# 数据集里会额外加 <bos>，因此输入长度是 seq_len + 1。
	block_size = 18
	dataset_seq_len = block_size - 1

	# 构建词表 & 数据集 & 数据加载
	tok = SimpleTokenizer()
	tok.build_vocab(texts)
	ds = CLMDataset(texts, tok, seq_len=dataset_seq_len)
	dl = DataLoader(ds, batch_size=4, shuffle=True)

	# 初始化、训练、生成
	device = "cuda" if torch.cuda.is_available() else "cpu"
	model = MiniGPT(len(tok.vocab), seq_len=block_size).to(device)
	train(model, dl, epochs=100, lr=1e-3, device=device)

	result = generate(model, tok, prompt="hello", max_len=20, strategy="top_k", top_k=5)
	print("生成结果： ", " ".join(result))


if __name__ == "__main__":
	main()
