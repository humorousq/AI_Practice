# Transformer 学习笔记

本目录包含Transformer架构的完整学习和实现过程。

## 学习路径

按照以下顺序学习，循序渐进地理解Transformer架构：

### 1️⃣ [注意力机制 (Attention Mechanism)](01_attention_mechanism/)
- Self-Attention的原理和实现
- Scaled Dot-Product Attention
- Multi-Head Attention
- 注意力机制的可视化

**关键概念**: Query, Key, Value, 注意力权重

### 2️⃣ [位置编码 (Position Encoding)](02_position_encoding/)
- 为什么需要位置编码
- 正弦/余弦位置编码
- 可学习的位置编码
- 位置编码的对比实验

**关键概念**: 序列顺序信息, 位置嵌入

### 3️⃣ [Transformer模块 (Transformer Blocks)](03_transformer_blocks/)
- Encoder Block的实现
- Decoder Block的实现
- Feed-Forward Network
- Layer Normalization
- Residual Connection

**关键概念**: 编码器-解码器架构, 残差连接, 层归一化

### 4️⃣ [完整模型 (Complete Model)](04_complete_model/)
- 完整Transformer模型的组装
- 训练流程实现
- 实际应用案例
- 模型调优技巧

**关键概念**: 端到端训练, 掩码机制, 模型应用

## 技术栈

- **Python 3.8+**
- **PyTorch** - 深度学习框架
- **NumPy** - 数值计算
- **Matplotlib** - 可视化

## 快速开始

```bash
# 安装依赖
pip install -r ../requirements.txt

# 运行某个模块的示例
cd 01_attention_mechanism
python self_attention.py
```

## 学习建议

1. **先理解再实现** - 每个模块先阅读README理解概念，再看代码实现
2. **动手实践** - 运行示例代码，修改参数观察效果
3. **循序渐进** - 按编号顺序学习，后面的内容依赖前面的基础
4. **做笔记** - 在README中记录自己的理解和疑问

## 参考资料

### 必读论文
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer原始论文

### 优秀教程
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - 图解Transformer
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) - 带注释的实现

### 官方文档
- [PyTorch Transformer](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html)

## 目录结构

```
transformer/
├── README.md                           # 本文件
├── 01_attention_mechanism/             # 注意力机制
│   ├── README.md
│   ├── self_attention.py
│   ├── multi_head_attention.py
│   └── test_attention.py
├── 02_position_encoding/               # 位置编码
│   ├── README.md
│   ├── positional_encoding.py
│   └── test_encoding.py
├── 03_transformer_blocks/              # Transformer块
│   ├── README.md
│   ├── encoder.py
│   ├── decoder.py
│   ├── feedforward.py
│   └── test_blocks.py
└── 04_complete_model/                  # 完整模型
    ├── README.md
    ├── transformer.py
    ├── train_example.py
    └── utils.py
```
