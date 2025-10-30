# Transformer模块 (Transformer Blocks)

## 概述

Transformer模块是将注意力机制、位置编码等组件组合起来的核心构建块。本章将实现完整的Encoder Block和Decoder Block，这些是构建完整Transformer模型的基础。

## 核心架构

### 1. Encoder Block (编码器块)

Encoder Block由以下组件组成，按顺序执行：

```
Input → Multi-Head Attention → Add & Norm → Feed Forward → Add & Norm → Output
```

**组件详解**：
1. **Multi-Head Self-Attention**: 处理输入序列的自注意力
2. **残差连接 + 层归一化**: 第一个Add & Norm
3. **Feed-Forward Network**: 位置相关的全连接网络
4. **残差连接 + 层归一化**: 第二个Add & Norm

### 2. Decoder Block (解码器块)

Decoder Block比Encoder多了一个masked self-attention层：

```
Input → Masked Multi-Head Attention → Add & Norm → 
Cross-Attention → Add & Norm → Feed Forward → Add & Norm → Output
```

**组件详解**：
1. **Masked Multi-Head Self-Attention**: 只能看到之前位置的信息
2. **残差连接 + 层归一化**: 第一个Add & Norm
3. **Multi-Head Cross-Attention**: Query来自decoder，Key和Value来自encoder
4. **残差连接 + 层归一化**: 第二个Add & Norm
5. **Feed-Forward Network**: 位置相关的全连接网络
6. **残差连接 + 层归一化**: 第三个Add & Norm

## 关键组件详解

### 1. Feed-Forward Network (FFN)

**结构**：
```
FFN(x) = max(0, xW1 + b1)W2 + b2
```

**特点**：
- 两层全连接网络
- 中间层使用ReLU激活函数
- 每个位置独立处理（位置相关）
- 中间层维度通常是模型维度的4倍

**作用**：
- 增加模型的非线性表达能力
- 为每个位置提供独立的变换

### 2. Layer Normalization (层归一化)

**公式**：
```
LayerNorm(x) = γ * (x - μ) / σ + β
```

**参数**：
- `μ`: 均值（沿特征维度计算）
- `σ`: 标准差（沿特征维度计算）
- `γ`: 可学习的缩放参数
- `β`: 可学习的偏移参数

**为什么用Layer Norm而不是Batch Norm？**
- **序列长度变化**: 不同batch的序列长度可能不同
- **更好的泛化**: 对batch size更鲁棒
- **推理一致性**: 训练和推理阶段行为一致

### 3. Residual Connection (残差连接)

**公式**：
```
Output = x + Sublayer(x)
```

**作用**：
- **解决梯度消失**: 提供梯度的直接路径
- **加速训练**: 让网络更容易学习恒等映射
- **提高稳定性**: 减少训练过程中的梯度爆炸

**Pre-Norm vs Post-Norm**：
- **Post-Norm** (原始论文): `LayerNorm(x + Sublayer(x))`
- **Pre-Norm** (更常用): `x + Sublayer(LayerNorm(x))`

Pre-Norm通常训练更稳定，收敛更快。

## Masking机制

### 1. Padding Mask (填充掩码)

用于处理不同长度的序列：
```python
# 对padding位置设置为-inf，经过softmax后变为0
mask = (input != pad_token).unsqueeze(1).unsqueeze(2)
scores = scores.masked_fill(mask == 0, -1e9)
```

### 2. Look-ahead Mask (前瞻掩码)

Decoder中使用，防止看到未来信息：
```python
# 下三角矩阵，只能看到当前和之前的位置
mask = torch.tril(torch.ones(seq_len, seq_len))
```

### 3. Cross-Attention Mask

在Encoder-Decoder attention中，通常只使用padding mask。

## 代码文件说明

### `encoder.py`
实现Encoder Block：
- EncoderLayer类
- Multi-Head Attention集成
- Feed-Forward Network
- 残差连接和层归一化

### `decoder.py`
实现Decoder Block：
- DecoderLayer类
- Masked Self-Attention
- Cross-Attention机制
- 完整的Decoder堆栈

### `feedforward.py`
实现Feed-Forward Network：
- 标准FFN实现
- 不同激活函数的支持
- 参数初始化策略

### `test_blocks.py`
测试和验证：
- 各个模块的功能测试
- 梯度流验证
- 性能基准测试

## 实现要点

### 1. 参数初始化

**Xavier/Glorot初始化**：
```python
# 对于线性层
nn.init.xavier_uniform_(self.weight)
```

**为什么重要？**
- 保持前向传播的方差稳定
- 避免梯度消失/爆炸
- 加速收敛

### 2. Dropout策略

**位置**：
- Attention权重后
- FFN激活函数后
- 残差连接前

**作用**：
- 防止过拟合
- 提高泛化能力
- 增加训练的鲁棒性

### 3. 维度管理

确保各层维度匹配：
- 输入维度: `[batch_size, seq_len, d_model]`
- 注意力输出: `[batch_size, seq_len, d_model]`
- FFN输出: `[batch_size, seq_len, d_model]`

## 关键要点

1. **模块化设计**: 每个组件独立实现，便于复用和测试
2. **残差连接的重要性**: 解决深度网络训练问题
3. **层归一化的作用**: 稳定训练过程
4. **掩码机制**: 控制信息流，实现不同的注意力模式
5. **Pre-Norm vs Post-Norm**: 影响训练稳定性

## 调试技巧

1. **检查维度**: 确保每一步的tensor维度正确
2. **验证掩码**: 可视化attention权重确认掩码生效
3. **梯度检查**: 确保梯度能正常反向传播
4. **小规模测试**: 先在小数据上验证正确性

## 练习建议

1. 实现基础的Encoder Block，测试前向传播
2. 添加Decoder Block，理解masked attention
3. 实验不同的层归一化位置（Pre-Norm vs Post-Norm）
4. 可视化注意力权重，观察模型关注的位置
5. 测试不同的FFN维度对性能的影响

## 常见问题

**Q: 为什么FFN的隐藏层维度要比输入大？**
A: 增加模型的表达能力，通常设为4倍可以在参数量和性能间取得好的平衡。

**Q: 残差连接为什么有效？**
A: 提供梯度的直接路径，让深层网络更容易训练，同时保留原始信息。

**Q: Layer Norm的位置为什么重要？**
A: Pre-Norm通常更稳定，Post-Norm是原始设计但训练可能不稳定。

## 下一步

学习完Transformer模块后，继续学习 [完整模型](../04_complete_model/)，了解如何将所有组件组装成完整的Transformer模型并进行训练。
