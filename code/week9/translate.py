import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
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

# 加载训练好的模型权重
checkpoint = torch.load('transformer_model.pth')

# 恢复词汇表和配置
word2idx_en = checkpoint['word2idx_en']
word2idx_zh = checkpoint['word2idx_zh']
idx2word_en = checkpoint['idx2word_en']
idx2word_zh = checkpoint['idx2word_zh']
max_len_en = checkpoint['max_len_en']
max_len_zh = checkpoint['max_len_zh']
model_config = checkpoint['model_config']

# 编码句子函数
def encode_sentence(sentence, tokenize_fn, word2idx, max_len):
    tokens = tokenize_fn(sentence)
    token_ids = [word2idx["<BOS>"]] + [word2idx.get(token, word2idx["<UNK>"]) for token in tokens] + [word2idx["<EOS>"]]
    if len(token_ids) < max_len:
        token_ids += [word2idx["<PAD>"]] * (max_len - len(token_ids))
    else:
        token_ids = token_ids[:max_len]
    return token_ids

# 重新实例化模型
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

# 加载模型权重
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("模型加载成功！")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~12、测试翻译过程~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def translate(model, src_sentence, max_len=20):
    model.eval()
    src_ids = encode_sentence(src_sentence, tokenize_en, word2idx_en, max_len_en)
    src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0)
    tgt_ids = [word2idx_zh["<BOS>"]]
    for i in range(max_len):
        tgt_tensor = torch.tensor(tgt_ids, dtype=torch.long).unsqueeze(0)
        tgt_mask = generate_square_subsequent_mask(tgt_tensor.size(1)).to(tgt_tensor.device).to(dtype=torch.float32)
        with torch.no_grad():
            output = model(src_tensor, tgt_tensor, tgt_mask=tgt_mask)
        next_token = output[0, -1, :].argmax().item()
        tgt_ids.append(next_token)
        if next_token == word2idx_zh["<EOS>"]:
            break
    # 过滤掉特殊标记 <BOS> 和 <EOS>
    translated_tokens = [idx2word_zh[idx] for idx in tgt_ids if idx2word_zh[idx] not in ["<BOS>", "<EOS>"]]
    return " ".join(translated_tokens)

# 测试翻译
test_sentences = ["I am a student.",
                  "He loves programming.",
                  "The weather is nice today.",
                  "Can you help me?",
                  "This is a test sentence."]

print("\n翻译测试：")
print("=" * 60)
for example_en in test_sentences:
    translated_result = translate(model, example_en)
    print(f"待翻译句子: {example_en}")
    print(f"翻译结果: {translated_result}")
    print("=" * 60)
