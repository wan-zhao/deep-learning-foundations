import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import json
import re
from collections import Counter

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~数据处理~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1. 从JSON文件加载平行翻译数据
def load_training_data(json_file='training_data.json'):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    parallel_data = [(item['en'], item['zh']) for item in data['parallel_data']]
    return parallel_data

# 2. 定义分词函数
def tokenize_en(sentence):
    sentence = sentence.lower()
    tokens = re.findall(r"[a-zA-Z]+|[,.!?']", sentence)
    return tokens

def tokenize_zh(sentence):
    return sentence.split()

# 3. 构建词汇表
def build_vocab(parallel_data):
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
    
    return word2idx_en, idx2word_en, word2idx_zh, idx2word_zh

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~自定义Dataset类~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class TranslationDataset(Dataset):
    """
    翻译数据集类
    
    参数:
        parallel_data: 平行语料列表，格式为 [(en_sentence, zh_sentence), ...]
        word2idx_en: 英文词到索引的映射
        word2idx_zh: 中文词到索引的映射
    """
    def __init__(self, parallel_data, word2idx_en, word2idx_zh):
        self.parallel_data = parallel_data
        self.word2idx_en = word2idx_en
        self.word2idx_zh = word2idx_zh
        
    def __len__(self):
        return len(self.parallel_data)
    
    def __getitem__(self, idx):
        """
        返回单个样本
        
        返回:
            dict: 包含源句子和目标句子的字典
                - 'src': 源句子的token id列表（不含padding）
                - 'tgt': 目标句子的token id列表（不含padding）
        """
        en_sentence, zh_sentence = self.parallel_data[idx]
        
        # 编码英文句子
        en_tokens = tokenize_en(en_sentence)
        src_ids = [self.word2idx_en["<BOS>"]] + \
                  [self.word2idx_en.get(token, self.word2idx_en["<UNK>"]) for token in en_tokens] + \
                  [self.word2idx_en["<EOS>"]]
        
        # 编码中文句子
        zh_tokens = tokenize_zh(zh_sentence)
        tgt_ids = [self.word2idx_zh["<BOS>"]] + \
                  [self.word2idx_zh.get(token, self.word2idx_zh["<UNK>"]) for token in zh_tokens] + \
                  [self.word2idx_zh["<EOS>"]]
        
        return {
            'src': torch.tensor(src_ids, dtype=torch.long),
            'tgt': torch.tensor(tgt_ids, dtype=torch.long)
        }

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Collate函数~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def collate_fn(batch, pad_idx_src=0, pad_idx_tgt=0):
    """
    批量数据的collate函数，用于DataLoader
    
    参数:
        batch: 一个batch的数据列表，每个元素是Dataset返回的dict
        pad_idx_src: 源语言的padding索引
        pad_idx_tgt: 目标语言的padding索引
    
    返回:
        dict: 包含批量数据的字典
            - 'src': (batch_size, max_src_len) 源句子tensor
            - 'tgt': (batch_size, max_tgt_len) 目标句子tensor
            - 'src_lengths': (batch_size,) 源句子实际长度
            - 'tgt_lengths': (batch_size,) 目标句子实际长度
    """
    # 提取源句子和目标句子
    src_batch = [item['src'] for item in batch]
    tgt_batch = [item['tgt'] for item in batch]
    
    # 记录原始长度
    src_lengths = torch.tensor([len(src) for src in src_batch], dtype=torch.long)
    tgt_lengths = torch.tensor([len(tgt) for tgt in tgt_batch], dtype=torch.long)
    
    # 使用pad_sequence进行padding
    # pad_sequence默认在序列末尾添加padding，batch_first=True使得返回形状为(batch_size, max_len)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx_src)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx_tgt)
    
    return {
        'src': src_padded,
        'tgt': tgt_padded,
        'src_lengths': src_lengths,
        'tgt_lengths': tgt_lengths
    }

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~创建DataLoader~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def create_dataloader(parallel_data, word2idx_en, word2idx_zh, batch_size=8, shuffle=True):
    """
    创建DataLoader
    
    参数:
        parallel_data: 平行语料列表
        word2idx_en: 英文词到索引的映射
        word2idx_zh: 中文词到索引的映射
        batch_size: 批量大小
        shuffle: 是否打乱数据
    
    返回:
        DataLoader对象
    """
    dataset = TranslationDataset(parallel_data, word2idx_en, word2idx_zh)
    
    # 创建collate_fn的偏函数，固定pad_idx参数
    from functools import partial
    collate_fn_with_pad = partial(
        collate_fn,
        pad_idx_src=word2idx_en["<PAD>"],
        pad_idx_tgt=word2idx_zh["<PAD>"]
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn_with_pad,
        num_workers=0  # Windows下建议设为0
    )
    
    return dataloader

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~使用示例~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    # 1. 加载数据
    print("加载训练数据...")
    parallel_data = load_training_data('training_data.json')
    print(f"数据集大小: {len(parallel_data)}")
    
    # 2. 构建词汇表
    print("\n构建词汇表...")
    word2idx_en, idx2word_en, word2idx_zh, idx2word_zh = build_vocab(parallel_data)
    print(f"英文词汇表大小: {len(word2idx_en)}")
    print(f"中文词汇表大小: {len(word2idx_zh)}")
    
    # 3. 创建DataLoader
    print("\n创建DataLoader...")
    batch_size = 8
    dataloader = create_dataloader(
        parallel_data,
        word2idx_en,
        word2idx_zh,
        batch_size=batch_size,
        shuffle=True
    )
    
    # 4. 测试DataLoader
    print(f"\n测试DataLoader (batch_size={batch_size}):")
    print("=" * 60)
    
    for batch_idx, batch in enumerate(dataloader):
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  源句子形状: {batch['src'].shape}")
        print(f"  目标句子形状: {batch['tgt'].shape}")
        print(f"  源句子长度: {batch['src_lengths'].tolist()}")
        print(f"  目标句子长度: {batch['tgt_lengths'].tolist()}")
        
        # 显示第一个样本的详细信息
        if batch_idx == 0:
            print(f"\n  第一个样本详情:")
            src_tokens = [idx2word_en[idx.item()] for idx in batch['src'][0] if idx.item() != word2idx_en["<PAD>"]]
            tgt_tokens = [idx2word_zh[idx.item()] for idx in batch['tgt'][0] if idx.item() != word2idx_zh["<PAD>"]]
            print(f"    源句子: {' '.join(src_tokens)}")
            print(f"    目标句子: {' '.join(tgt_tokens)}")
        
        # 只显示前3个batch
        if batch_idx >= 2:
            break
    
    print("\n" + "=" * 60)
    print("DataLoader测试完成！")
    
    # 5. 展示如何在训练循环中使用
    print("\n训练循环示例:")
    print("-" * 60)
    print("""
    for epoch in range(num_epochs):
        for batch in dataloader:
            src = batch['src']  # (batch_size, max_src_len)
            tgt = batch['tgt']  # (batch_size, max_tgt_len)
            
            # 准备decoder输入和输出
            tgt_input = tgt[:, :-1]  # 去掉最后一个token
            tgt_output = tgt[:, 1:]  # 去掉第一个token (<BOS>)
            
            # 生成mask
            tgt_seq_len = tgt_input.size(1)
            tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
            
            # 前向传播
            logits = model(src, tgt_input, tgt_mask=tgt_mask)
            
            # 计算损失
            loss = criterion(logits.reshape(-1, vocab_size), tgt_output.reshape(-1))
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    """)
    print("-" * 60)
