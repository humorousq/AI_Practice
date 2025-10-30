# 完整Transformer模型 (Complete Model)

## 概述

本章将所有前面学习的组件组装成完整的Transformer模型，包括模型构建、训练流程、实际应用案例以及性能优化技巧。

## 完整架构

### Transformer整体结构

```
输入序列 → Input Embedding + Position Encoding
    ↓
Encoder Stack (N个Encoder Blocks)
    ↓
目标序列 → Output Embedding + Position Encoding
    ↓
Decoder Stack (N个Decoder Blocks) ← Encoder输出
    ↓
Linear + Softmax → 输出概率分布
```

**核心参数**：
- `d_model`: 模型维度（通常512或768）
- `N`: Encoder/Decoder层数（通常6层）
- `h`: 注意力头数（通常8）
- `d_ff`: FFN隐藏层维度（通常2048）
- `vocab_size`: 词汇表大小
- `max_seq_length`: 最大序列长度

## 模型组件详解

### 1. 输入嵌入层 (Input Embeddings)

**Token Embedding**：
```python
self.token_embedding = nn.Embedding(vocab_size, d_model)
```

**位置编码**：
```python
# 固定的正弦位置编码
pe = torch.zeros(max_len, d_model)
position = torch.arange(0, max_len).unsqueeze(1).float()
div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                    -(math.log(10000.0) / d_model))
pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)
```

**组合方式**：
```python
embedded = self.token_embedding(x) * math.sqrt(self.d_model) + self.pos_encoding
```

**为什么要乘以√d_model？**
- 平衡token embedding和position encoding的尺度
- 防止position encoding被token embedding淹没

### 2. 输出层 (Output Layer)

**线性投影**：
```python
self.output_projection = nn.Linear(d_model, vocab_size)
```

**Softmax**：
```python
# 在训练时通常在loss函数中计算
output_probs = F.softmax(logits, dim=-1)
```

**权重共享**：
- 输入embedding和输出投影可以共享权重
- 减少参数量，提高泛化能力

### 3. 完整前向传播

**Encoder-Decoder流程**：
```python
def forward(self, src, tgt, src_mask=None, tgt_mask=None):
    # Encoder
    src_embedded = self.input_embedding(src) + self.pos_encoding
    encoder_output = self.encoder(src_embedded, src_mask)
    
    # Decoder
    tgt_embedded = self.output_embedding(tgt) + self.pos_encoding
    decoder_output = self.decoder(tgt_embedded, encoder_output, 
                                 tgt_mask, src_mask)
    
    # Output projection
    return self.output_projection(decoder_output)
```

## 训练流程

### 1. 数据预处理

**分词 (Tokenization)**：
```python
# 使用BPE或WordPiece等子词算法
tokenizer = BPETokenizer()
tokens = tokenizer.encode(text)
```

**序列处理**：
- 添加特殊token：`<BOS>`, `<EOS>`, `<PAD>`, `<UNK>`
- 序列截断或填充到固定长度
- 创建attention mask

### 2. 损失函数

**交叉熵损失**：
```python
criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
```

**标签平滑 (Label Smoothing)**：
```python
class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        
    def forward(self, pred, target):
        # 实现标签平滑逻辑
        pass
```

**为什么使用标签平滑？**
- 防止过度自信的预测
- 提高模型泛化能力
- 减少过拟合

### 3. 优化器设置

**Adam优化器**：
```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.98),
    eps=1e-9
)
```

**学习率调度**：
```python
# Transformer原论文中的warmup策略
lr = d_model ** -0.5 * min(step ** -0.5, step * warmup_steps ** -1.5)
```

**为什么需要warmup？**
- 防止训练初期梯度过大
- 让模型逐渐适应训练过程
- 提高训练稳定性

### 4. 训练循环

```python
def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        # 前向传播
        logits = model(batch.src, batch.tgt[:-1])
        loss = criterion(logits.reshape(-1, vocab_size), 
                        batch.tgt[1:].reshape(-1))
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)
```

**梯度裁剪的重要性**：
- 防止梯度爆炸
- 提高训练稳定性
- 特别重要在序列模型中

## 推理 (Inference)

### 1. 贪心解码 (Greedy Decoding)

```python
def greedy_decode(model, src, max_len=100):
    model.eval()
    
    # Encoder
    encoder_output = model.encode(src)
    
    # Decoder逐步生成
    decoded = [BOS_token]
    for _ in range(max_len):
        tgt_tensor = torch.tensor(decoded).unsqueeze(0)
        with torch.no_grad():
            logits = model.decode(tgt_tensor, encoder_output)
            next_token = logits.argmax(dim=-1)[-1].item()
        
        if next_token == EOS_token:
            break
        decoded.append(next_token)
    
    return decoded
```

### 2. 束搜索 (Beam Search)

```python
def beam_search(model, src, beam_size=5, max_len=100):
    # 维护beam_size个候选序列
    # 每步扩展所有候选，保留概率最高的beam_size个
    pass
```

**Beam Search优点**：
- 比贪心搜索找到更好的序列
- 可控的计算复杂度

**超参数选择**：
- `beam_size`: 通常3-10
- 更大的beam不一定更好
- 需要在质量和速度间平衡

### 3. 采样策略

**Top-k采样**：
```python
# 只从概率最高的k个token中采样
top_k_logits = torch.topk(logits, k=50).values
```

**Top-p采样 (Nucleus Sampling)**：
```python
# 从累积概率达到p的最小集合中采样
sorted_probs = torch.sort(probs, descending=True)
cumulative_probs = torch.cumsum(sorted_probs.values, dim=-1)
```

## 实际应用案例

### 1. 机器翻译

**数据集**: WMT14 En-De
**任务**: 英语 → 德语翻译
**评估指标**: BLEU分数

**特定设置**：
- 共享源语言和目标语言的词汇表
- 使用BPE进行子词分割
- 序列长度通常256-512

### 2. 文本摘要

**数据集**: CNN/DailyMail
**任务**: 新闻文章 → 摘要
**评估指标**: ROUGE分数

**特定技巧**：
- 源序列较长（1024+）
- 目标序列较短（100-200）
- 可能需要位置编码的特殊处理

### 3. 对话系统

**数据集**: PersonaChat, MultiWOZ
**任务**: 上下文 → 回复生成
**评估指标**: 困惑度、人工评估

**特殊考虑**：
- 多轮对话历史的编码
- 人格一致性
- 回复的多样性

## 模型优化技巧

### 1. 内存优化

**梯度累积**：
```python
# 模拟更大的batch size
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**混合精度训练**：
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    loss = model(batch)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 2. 计算优化

**Flash Attention**：
- 优化attention计算的内存使用
- 显著减少GPU内存占用
- 加速训练和推理

**模型并行**：
- 在多GPU上分布模型参数
- 处理超大模型（GPT-3规模）

### 3. 训练技巧

**课程学习 (Curriculum Learning)**：
- 从简单样本开始训练
- 逐渐增加样本难度
- 提高收敛速度和最终性能

**数据增强**：
- Back-translation
- Paraphrasing
- Noise injection

## 代码文件说明

### `transformer.py`
完整的Transformer模型实现：
- 整合所有组件
- 支持自定义参数配置
- 包含训练和推理接口

### `train_example.py`
完整的训练示例：
- 数据加载和预处理
- 训练循环实现
- 模型保存和加载
- 性能监控和日志

### `utils.py`
辅助工具函数：
- 掩码生成
- 学习率调度
- 评估指标计算
- 数据处理工具

## 评估指标

### 1. 机器翻译
- **BLEU**: 测量生成文本与参考文本的n-gram重叠
- **chrF**: 基于字符级的评估
- **BERTScore**: 基于语义相似度

### 2. 文本生成
- **困惑度 (Perplexity)**: 模型对测试数据的预测能力
- **ROUGE**: 摘要任务的标准评估
- **人工评估**: 流畅性、相关性、事实性

### 3. 效率指标
- **推理速度**: tokens/second
- **内存使用**: 峰值GPU内存
- **参数量**: 模型大小

## 常见问题和解决方案

### 1. 训练不稳定
**症状**: 损失震荡、梯度爆炸
**解决**: 梯度裁剪、学习率调整、预热

### 2. 生成重复
**症状**: 模型生成重复的文本片段
**解决**: 束搜索、重复惩罚、采样策略

### 3. 内存不足
**症状**: CUDA out of memory
**解决**: 减小batch size、梯度累积、混合精度

## 扩展阅读

### 模型变种
- **BERT**: 只使用Encoder的双向模型
- **GPT**: 只使用Decoder的自回归模型
- **T5**: Text-to-Text Transfer Transformer
- **PaLM**: 超大规模语言模型

### 最新发展
- **Transformer-XL**: 处理长序列
- **Reformer**: 减少内存复杂度
- **Longformer**: 稀疏注意力机制
- **Switch Transformer**: 专家混合模型

## 项目实践建议

1. **从小做起**: 先在小数据集上验证模型正确性
2. **逐步扩展**: 增加模型大小和数据规模
3. **监控训练**: 实时跟踪损失、梯度、学习率
4. **对比基线**: 与现有模型比较性能
5. **消融实验**: 验证各个组件的贡献

## 总结

通过学习完整的Transformer模型，你已经掌握了：

1. **架构理解**: 从注意力机制到完整模型的演进
2. **实现能力**: 从零开始构建Transformer
3. **训练技巧**: 高效训练大规模模型的方法
4. **应用知识**: 将模型应用到实际NLP任务
5. **优化经验**: 提高模型性能和效率的技巧

这些知识为你继续探索更高级的NLP模型（如BERT、GPT等）奠定了坚实的基础。记住，理论学习和实践相结合是掌握深度学习的关键！
