import re
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Tokenizer 和输入编码器~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
def basic_tokenizer(text):
    # 小写 + 保留基本符号
    text = text.lower()
    tokens = re.findall(r"[a-z]+|[.,!?;]", text)
    return tokens
# 简单词表
vocab = ['[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]'] + [
'i', 'like', 'cats', 'dogs', 'you', 'do', 'not', 'know', 'love', 'playing', 'football',
'.', '?'
]
token2id = {word: idx for idx, word in enumerate(vocab)}
id2token = {idx: word for word, idx in token2id.items()}
def encode(tokens, max_len=12):
    token_ids = [token2id.get(t, token2id['[UNK]']) for t in tokens]
    token_ids = [token2id['[CLS]']] + token_ids + [token2id['[SEP]']]
    return pad_or_truncate(token_ids, pad_value=token2id['[PAD]'], max_len=max_len)
def pad_or_truncate(seq, pad_value, max_len):
    return seq[:max_len] + [pad_value] * (max_len - len(seq))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~MLM+NSP数据生成器（模拟语料）~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
import random
import torch
from torch.utils.data import Dataset
class BertPretrainDataset(Dataset):
    def __init__(self, sentence_pairs, tokenizer, mask_prob=0.15, max_len=12):
        self.pairs = sentence_pairs # [(A, B), (A, random B)]
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
        self.max_len = max_len
    def __len__(self):
        return len(self.pairs)
    def random_mask(self, token_ids):
        output = token_ids.copy()
        labels = [-100] * len(token_ids)
        for i in range(1, len(token_ids) - 1):
            if token_ids[i] == token2id['[PAD]']:
                continue
            if random.random() < self.mask_prob:
                labels[i] = token_ids[i]
                prob = random.random()
                if prob < 0.8:
                    output[i] = token2id['[MASK]']
                elif prob < 0.9:
                    output[i] = random.randint(5, len(vocab) - 1)
        return output, labels
    def __getitem__(self, idx):
        A, B, is_next = self.pairs[idx]
        tokens_A = basic_tokenizer(A)
        tokens_B = basic_tokenizer(B)
        # 拼接、编码 + 截断
        all_tokens = tokens_A + ['[SEP]'] + tokens_B
        input_ids = encode(all_tokens, max_len=self.max_len)
        # segment_ids 构造： 0 for sentence A, 1 for sentence B
        len_A = len(tokens_A) + 2 # 包括 [CLS] 和第一个 [SEP]
        len_B = len(input_ids) - len_A
        segment_ids = [0] * len_A + [1] * len_B
        segment_ids = pad_or_truncate(segment_ids, pad_value=0, max_len=self.max_len)
        # MLM 掩码
        input_ids_masked, mlm_labels = self.random_mask(input_ids)
        input_ids_masked = pad_or_truncate(input_ids_masked, pad_value=token2id['[PAD]'],
max_len=self.max_len)
        mlm_labels = pad_or_truncate(mlm_labels, pad_value=-100, max_len=self.max_len)
        return {
            'input_ids': torch.tensor(input_ids_masked),
            'token_type_ids': torch.tensor(segment_ids),
            'mlm_labels': torch.tensor(mlm_labels),
            'nsp_label': torch.tensor(1 if is_next else 0)
        }
# 构造句对
sentence_pool = [
"I like cats.", "Do you like dogs?", "I do not know.", "He is playing football.", "I love you."]
sentence_pairs = []
for i in range(len(sentence_pool) - 1):
    sentence_pairs.append((sentence_pool[i], sentence_pool[i+1], True))
    sentence_pairs.append((sentence_pool[i], random.choice(sentence_pool), False))
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~BERT模型定义~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
import torch.nn as nn
class TinyBert(nn.Module):
    def __init__(self, vocab_size, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.segment_embedding = nn.Embedding(2, hidden_dim)
        self.position_embedding = nn.Embedding(12, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=2)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.mlm_head = nn.Linear(hidden_dim, vocab_size)
        self.nsp_head = nn.Linear(hidden_dim, 2)
    def forward(self, input_ids, token_type_ids):
        position_ids = torch.arange(input_ids.size(1)).unsqueeze(0).to(input_ids.device)
        x = self.embedding(input_ids) + \
            self.segment_embedding(token_type_ids) + \
            self.position_embedding(position_ids)
        x = self.encoder(x)
        mlm_logits = self.mlm_head(x)
        cls_rep = x[:, 0, :] # [CLS]
        nsp_logits = self.nsp_head(cls_rep)
        return mlm_logits, nsp_logits
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~训练循环~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
from torch.utils.data import DataLoader
model = TinyBert(vocab_size=len(vocab))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_mlm = nn.CrossEntropyLoss(ignore_index=-100)
loss_nsp = nn.CrossEntropyLoss()
dataloader = DataLoader(BertPretrainDataset(sentence_pairs, basic_tokenizer), batch_size=4,
shuffle=True)

for epoch in range(5):
    model.train()
    for batch in dataloader:
        mlm_logits, nsp_logits = model(batch['input_ids'], batch['token_type_ids'])
        mlm_loss = loss_mlm(mlm_logits.view(-1, len(vocab)), batch['mlm_labels'].view(-1))
        nsp_loss = loss_nsp(nsp_logits, batch['nsp_label'])
        loss = mlm_loss + nsp_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")