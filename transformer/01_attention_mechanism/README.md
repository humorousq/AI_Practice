# 注意力机制 (Attention Mechanism)

## 概述

注意力机制是Transformer的核心组件，它允许模型在处理序列时关注不同位置的信息。

## 核心概念

### 1. Self-Attention (自注意力)

Self-Attention允许序列中的每个位置关注序列中的所有位置，从而捕获序列内部的依赖关系。

**基本思想**：
- 对于输入序列中的每个元素，计算它与序列中所有元素的相关性
- 基于相关性对所有元素进行加权求和

### 2. Scaled Dot-Product Attention

**公式**：
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**参数说明**：
- **Q (Query)**: 查询向量，表示"我在找什么"
- **K (Key)**: 键向量，表示"我是什么"
- **V (Value)**: 值向量，表示"我的内容是什么"
- **d_k**: Key向量的维度，用于缩放防止梯度消失

**计算步骤**：
1. 计算Query和Key的点积：`QK^T`
2. 除以√d_k进行缩放
3. 应用softmax得到注意力权重
4. 用权重对Value进行加权求和

### 3. Multi-Head Attention (多头注意力)

**为什么需要多头？**
- 单个注意力头可能只关注某一方面的特征
- 多个头可以并行地关注不同的特征子空间

**公式**：
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O

其中 head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

**参数**：
- **h**: 注意力头的数量（通常为8）
- **W^Q, W^K, W^V**: 每个头的投影矩阵
- **W^O**: 输出投影矩阵

## 代码文件说明

### `self_attention.py`
实现基础的Self-Attention机制：
- Scaled Dot-Product Attention
- 注意力权重的计算和可视化

### `multi_head_attention.py`
实现Multi-Head Attention：
- 多头注意力的完整实现
- 参数初始化和前向传播

### `test_attention.py`
测试和验证代码：
- 测试注意力机制的正确性
- 可视化注意力权重
- 简单示例演示

## 关键要点

1. **注意力机制的本质**：动态地为输入序列的不同部分分配不同的权重
2. **Scaling的重要性**：防止点积过大导致softmax梯度消失
3. **多头的优势**：捕获多种不同类型的依赖关系
4. **计算复杂度**：O(n²d)，n是序列长度，d是特征维度

## 练习建议

1. 实现并运行基础的Self-Attention
2. 修改head数量，观察多头注意力的效果
3. 可视化注意力权重矩阵，理解模型关注的位置
4. 尝试不同的缩放因子，观察对训练的影响

## 下一步

学习完注意力机制后，继续学习 [位置编码](../02_position_encoding/)，了解如何为序列添加位置信息。
