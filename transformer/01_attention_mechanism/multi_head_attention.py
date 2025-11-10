import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制模块
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 定义 Q, K, V 和输出的线性变换层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # 1. 计算注意力得分 (QK^T)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 2. 应用掩码 (如果提供)
        if mask is not None:
            # 将掩码中为 0 的位置设置为一个非常小的负数，这样 softmax 后会接近 0
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # 3. 计算注意力权重 (Softmax)
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # 4. 加权求和 (权重 * V)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        # 将输入 x 的形状从 (batch_size, seq_length, d_model)
        # 变换为 (batch_size, num_heads, seq_length, d_k)
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        # 将输入 x 的形状从 (batch_size, num_heads, seq_length, d_k)
        # 变回 (batch_size, seq_length, d_model)
        batch_size, num_heads, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        # 1. 对 Q, K, V 进行线性变换
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        # 2. 计算缩放点积注意力
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # 3. 合并多头输出并进行最终的线性变换
        output = self.W_o(self.combine_heads(attn_output))
        return output


if __name__ == "__main__":
    # 演示 torch.nn 的使用
    print("=== torch.nn 模块演示 ===")
    
    # 1. 基本参数设置
    batch_size = 2
    seq_length = 4
    d_model = 8
    num_heads = 2
    
    # 2. 创建多头注意力模块 (继承自 nn.Module)
    attention = MultiHeadAttention(d_model, num_heads)
    
    # 3. 查看模型参数 (nn.Module 自动管理)
    print(f"模型参数数量: {sum(p.numel() for p in attention.parameters())}")
    print(f"可训练参数: {sum(p.numel() for p in attention.parameters() if p.requires_grad)}")
    
    # 4. 查看模型结构
    print("\n模型结构:")
    for name, module in attention.named_modules():
        if name:  # 跳过根模块
            print(f"  {name}: {module}")
    
    # 5. 创建示例输入
    x = torch.randn(batch_size, seq_length, d_model)
    print(f"\n输入形状: {x.shape}")
    
    # 6. 前向传播 (nn.Module 自动调用 forward 方法)
    output = attention(x, x, x)  # Self-attention: Q=K=V=x
    print(f"输出形状: {output.shape}")
    
    # 7. 演示 nn.Module 的其他功能
    print(f"\n模型当前模式: {'训练' if attention.training else '评估'}")
    
    # 切换到评估模式
    attention.eval()
    print(f"切换后模式: {'训练' if attention.training else '评估'}")
    
    # 8. 演示参数访问
    print("\n线性层参数:")
    for name, param in attention.named_parameters():
        print(f"  {name}: {param.shape}")
    
    # 9. 演示 torch.nn 中的其他常用组件
    print("\n=== torch.nn 其他常用组件 ===")
    
    # 激活函数
    relu = nn.ReLU()
    softmax = nn.Softmax(dim=-1)
    
    # 正则化
    dropout = nn.Dropout(0.1)
    layer_norm = nn.LayerNorm(d_model)
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 示例：组合使用
    x_norm = layer_norm(x)
    x_dropout = dropout(x_norm)
    
    print(f"LayerNorm 后形状: {x_norm.shape}")
    print(f"Dropout 后形状: {x_dropout.shape}")
    
    print("\n✅ torch.nn 演示完成!")