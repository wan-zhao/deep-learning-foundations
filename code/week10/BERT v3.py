import math
import random
import re
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 尽量使用更真实的分词器：优先用 HuggingFace 的 BertTokenizerFast，否则回退到简易子词分词
try:
    from transformers import BertTokenizerFast  # type: ignore
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False


##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Tokenizer 和 vocab~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##

SPECIAL_TOKENS = {
    'pad': '[PAD]',
    'cls': '[CLS]',
    'sep': '[SEP]',
    'mask': '[MASK]',
    'unk': '[UNK]'
}


def build_fallback_vocab(texts: List[str], min_freq: int = 1) -> List[str]:
    freq = {}
    for t in texts:
        for tok in re.findall(r"[a-z]+|[.,!?;]", t.lower()):
            freq[tok] = freq.get(tok, 0) + 1
    vocab = list(SPECIAL_TOKENS.values())
    vocab += [tok for tok, c in freq.items() if c >= min_freq]
    return vocab


class SimpleWPTokenizer:
    """简易子词风格分词，作为 BertTokenizerFast 的备选。"""

    def __init__(self, texts: List[str], max_len: int = 64):
        self.vocab_list = build_fallback_vocab(texts)
        self.token2id = {w: i for i, w in enumerate(self.vocab_list)}
        self.id2token = {i: w for w, i in self.token2id.items()}
        self.pad_token = SPECIAL_TOKENS['pad']
        self.cls_token = SPECIAL_TOKENS['cls']
        self.sep_token = SPECIAL_TOKENS['sep']
        self.mask_token = SPECIAL_TOKENS['mask']
        self.unk_token = SPECIAL_TOKENS['unk']
        self.pad_token_id = self.token2id[self.pad_token]
        self.cls_token_id = self.token2id[self.cls_token]
        self.sep_token_id = self.token2id[self.sep_token]
        self.mask_token_id = self.token2id[self.mask_token]
        self.unk_token_id = self.token2id[self.unk_token]
        self.max_len = max_len

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-z]+|[.,!?;]", text.lower())

    def encode_plus(self, text: str, text_pair: str = None, max_length: int = None,
                    padding: str = 'max_length', truncation: bool = True, return_tensors=None):
        max_len = max_length or self.max_len
        tokens_a = self._tokenize(text)
        tokens_b = self._tokenize(text_pair) if text_pair is not None else []
        tokens = [self.cls_token] + tokens_a + [self.sep_token]
        token_type_ids = [0] * len(tokens)
        if tokens_b:
            tokens += tokens_b + [self.sep_token]
            token_type_ids += [1] * (len(tokens_b) + 1)
        # 转 id
        input_ids = [self.token2id.get(t, self.unk_token_id) for t in tokens]
        # 截断
        if truncation and len(input_ids) > max_len:
            input_ids = input_ids[:max_len]
            token_type_ids = token_type_ids[:max_len]
        # padding
        pad_len = max_len - len(input_ids)
        if padding == 'max_length' and pad_len > 0:
            input_ids += [self.pad_token_id] * pad_len
            token_type_ids += [0] * pad_len
        attention_mask = [1 if i != self.pad_token_id else 0 for i in input_ids]
        out = {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask
        }
        return out

    def get_vocab(self):
        return self.token2id

    @property
    def vocab_size(self):
        return len(self.vocab_list)


def load_tokenizer(texts: List[str], max_len: int = 64):
    """优先使用 bert-base-uncased；缺失则退回简单分词器。"""
    if HF_AVAILABLE:
        try:
            tok = BertTokenizerFast.from_pretrained('bert-base-uncased')
            tok.model_max_length = max_len
            return tok, tok.vocab_size
        except Exception:
            pass
    fallback = SimpleWPTokenizer(texts, max_len=max_len)
    return fallback, fallback.vocab_size


##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~MLM + NSP Dataset~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##

class BertPretrainDataset(Dataset):
    def __init__(self, sentence_pairs: List[Tuple[str, str, bool]], tokenizer, mask_prob: float = 0.15,
                 max_len: int = 64):
        self.pairs = sentence_pairs
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
        self.max_len = max_len
        # id 获取
        self.pad_id = getattr(tokenizer, 'pad_token_id', tokenizer.token2id[SPECIAL_TOKENS['pad']])
        self.mask_id = getattr(tokenizer, 'mask_token_id', tokenizer.token2id[SPECIAL_TOKENS['mask']])
        self.cls_id = getattr(tokenizer, 'cls_token_id', tokenizer.token2id[SPECIAL_TOKENS['cls']])
        self.sep_id = getattr(tokenizer, 'sep_token_id', tokenizer.token2id[SPECIAL_TOKENS['sep']])

    def __len__(self):
        return len(self.pairs)

    def random_mask(self, input_ids: List[int]):
        output = input_ids.copy()
        labels = [-100] * len(input_ids)
        for i in range(1, len(input_ids) - 1):  # 跳过 CLS 和最后一位（通常是 SEP/PAD）
            tok_id = input_ids[i]
            if tok_id == self.pad_id:
                continue
            if random.random() < self.mask_prob:
                labels[i] = tok_id
                prob = random.random()
                if prob < 0.8:
                    output[i] = self.mask_id
                elif prob < 0.9:
                    # 随机词，但避免特殊符号区间
                    output[i] = random.randint(5, self.tokenizer.vocab_size - 1)
                else:
                    output[i] = tok_id
        return output, labels

    def __getitem__(self, idx):
        A, B, is_next = self.pairs[idx]
        encoded = self.tokenizer.encode_plus(A, B, max_length=self.max_len, padding='max_length', truncation=True)
        input_ids = encoded['input_ids']
        token_type_ids = encoded['token_type_ids']
        attention_mask = encoded['attention_mask']
        # 动态 MLM
        input_ids_masked, mlm_labels = self.random_mask(input_ids)
        return {
            'input_ids': torch.tensor(input_ids_masked, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'mlm_labels': torch.tensor(mlm_labels, dtype=torch.long),
            'nsp_label': torch.tensor(1 if is_next else 0, dtype=torch.long)
        }


##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~句对构造~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
sentence_pool = [
    "I like cats.",
    "Do you like dogs?",
    "I do not know.",
    "He is playing football.",
    "I love you.",
    "She enjoys painting in the evening.",
    "The sun is bright today.",
    "Reading books helps me relax.",
    "We are cooking dinner together.",
    "Music makes the day better.",
    "Artificial intelligence is changing the world.",
    "Learning new skills is always beneficial.",
    "Please send me the report by Friday.",
    "The project deadline is approaching fast.",
    "Have you seen the latest movie?",
    "My phone battery is almost dead.",
    "What time is the meeting tomorrow?",
    "The train was delayed by thirty minutes.",
    "This restaurant has amazing food.",
    "Let us grab coffee next week."
]

sentence_pairs: List[Tuple[str, str, bool]] = []
for i in range(len(sentence_pool) - 1):
    sentence_pairs.append((sentence_pool[i], sentence_pool[i + 1], True))
    sentence_pairs.append((sentence_pool[i], random.choice(sentence_pool), False))


##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~BERT 模型定义~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##

class TinyBert(nn.Module):
    def __init__(self, vocab_size: int, max_len: int = 64, hidden_dim: int = 128,
                 n_layers: int = 4, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.segment_embedding = nn.Embedding(2, hidden_dim)
        self.position_embedding = nn.Embedding(max_len, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.mlm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.nsp_head = nn.Linear(hidden_dim, 2)
        # 权重 tying
        self.mlm_head.weight = self.embedding.weight

    def forward(self, input_ids, token_type_ids, attention_mask=None):
        position_ids = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        x = self.embedding(input_ids) + \
            self.segment_embedding(token_type_ids) + \
            self.position_embedding(position_ids)
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        mlm_logits = self.mlm_head(x)
        cls_rep = x[:, 0, :]  # [CLS]
        nsp_logits = self.nsp_head(cls_rep)
        return mlm_logits, nsp_logits


##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~训练循环~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train():
    max_len = 64
    tokenizer, vocab_size = load_tokenizer(sentence_pool, max_len=max_len)
    ds = BertPretrainDataset(sentence_pairs, tokenizer, mask_prob=0.15, max_len=max_len)
    dl = DataLoader(ds, batch_size=8, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TinyBert(vocab_size=vocab_size, max_len=max_len, hidden_dim=128, n_layers=4, n_heads=4).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    total_steps = max(1, len(dl) * 10)
    warmup_steps = max(10, int(total_steps * 0.1))

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, (total_steps - current_step) / float(max(1, total_steps - warmup_steps)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    loss_mlm = nn.CrossEntropyLoss(ignore_index=-100)
    loss_nsp = nn.CrossEntropyLoss()

    epochs = 10
    global_step = 0
    for epoch in range(epochs):
        model.train()
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            mlm_logits, nsp_logits = model(
                batch['input_ids'], batch['token_type_ids'], batch['attention_mask']
            )
            mlm_loss = loss_mlm(mlm_logits.view(-1, vocab_size), batch['mlm_labels'].view(-1))
            nsp_loss = loss_nsp(nsp_logits, batch['nsp_label'])
            loss = mlm_loss + nsp_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")


if __name__ == "__main__":
    set_seed(42)
    train()
