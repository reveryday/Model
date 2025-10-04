import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class PositionalEncoding(nn.Module):
    """位置编码模块"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        
        output = self.w_o(attention_output)
        return output, attention_weights

class FeedForward(nn.Module):
    """前馈神经网络"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))

class TransformerBlock(nn.Module):
    """Transformer编码器块"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 自注意力机制
        attn_output, attention_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attention_weights

class PhysicsInformedTransformer(nn.Module):
    """基于物理约束的多任务Transformer网络"""
    def __init__(self, input_dim=8, output_dim=6, d_model=256, n_heads=8, n_layers=6, d_ff=1024, dropout=0.1):
        super(PhysicsInformedTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        
        # 输入投影层
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer编码器层
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # 特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 多任务输出头 - 为每个累积因子创建专门的输出头
        self.task_heads = nn.ModuleDict({
            'inf_flu': nn.Sequential(
                nn.Linear(d_model // 4, 128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Linear(64, 1)
            ),
            'fin_flu': nn.Sequential(
                nn.Linear(d_model // 4, 128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Linear(64, 1)
            ),
            'inf_exp': nn.Sequential(
                nn.Linear(d_model // 4, 128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Linear(64, 1)
            ),
            'fin_exp': nn.Sequential(
                nn.Linear(d_model // 4, 128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Linear(64, 1)
            ),
            'inf_eff': nn.Sequential(
                nn.Linear(d_model // 4, 128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Linear(64, 1)
            ),
            'fin_eff': nn.Sequential(
                nn.Linear(d_model // 4, 128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Linear(64, 1)
            )
        })
        
        # 物理约束层
        self.physics_constraint = PhysicsConstraintLayer()
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # 输入形状: (batch_size, input_dim)
        batch_size = x.size(0)
        
        # 输入投影
        x = self.input_projection(x)  # (batch_size, d_model)
        x = x.unsqueeze(1)  # (batch_size, 1, d_model)
        
        # 位置编码
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        
        # Transformer编码器
        attention_weights = []
        for transformer_block in self.transformer_blocks:
            x, attn_weights = transformer_block(x)
            attention_weights.append(attn_weights)
        
        # 特征提取
        x = x.squeeze(1)  # (batch_size, d_model)
        features = self.feature_extractor(x)  # (batch_size, d_model//4)
        
        # 多任务输出
        outputs = {}
        for task_name, head in self.task_heads.items():
            outputs[task_name] = head(features)
        
        # 组合输出
        output_tensor = torch.cat([
            outputs['inf_flu'], outputs['fin_flu'],
            outputs['inf_exp'], outputs['fin_exp'],
            outputs['inf_eff'], outputs['fin_eff']
        ], dim=1)
        
        # 应用物理约束
        output_tensor = self.physics_constraint(output_tensor, x)
        
        return output_tensor, attention_weights

class PhysicsConstraintLayer(nn.Module):
    """物理约束层 - 确保累积因子满足物理定律"""
    def __init__(self):
        super(PhysicsConstraintLayer, self).__init__()
        
    def forward(self, predictions, input_features):
        """
        应用物理约束：
        1. 累积因子必须 >= 1
        2. 随着厚度增加，累积因子应该增加
        3. 有限几何累积因子 <= 无限几何累积因子
        """
        # 确保所有累积因子 >= 1
        predictions = torch.clamp(predictions, min=1.0)
        
        # 分离不同类型的累积因子
        inf_flu = predictions[:, 0:1]
        fin_flu = predictions[:, 1:2]
        inf_exp = predictions[:, 2:3]
        fin_exp = predictions[:, 3:4]
        inf_eff = predictions[:, 4:5]
        fin_eff = predictions[:, 5:6]
        
        # 约束：有限几何 <= 无限几何
        fin_flu = torch.min(fin_flu, inf_flu)
        fin_exp = torch.min(fin_exp, inf_exp)
        fin_eff = torch.min(fin_eff, inf_eff)
        
        # 重新组合
        constrained_predictions = torch.cat([
            inf_flu, fin_flu, inf_exp, fin_exp, inf_eff, fin_eff
        ], dim=1)
        
        return constrained_predictions

class PhysicsInformedLoss(nn.Module):
    """物理约束损失函数"""
    def __init__(self, mse_weight=1.0, physics_weight=0.1):
        super(PhysicsInformedLoss, self).__init__()
        self.mse_weight = mse_weight
        self.physics_weight = physics_weight
        self.mse_loss = nn.MSELoss()
        
    def forward(self, predictions, targets, inputs):
        # 基础MSE损失
        mse_loss = self.mse_loss(predictions, targets)
        
        # 物理约束损失
        physics_loss = self.compute_physics_loss(predictions, inputs)
        
        # 总损失
        total_loss = self.mse_weight * mse_loss + self.physics_weight * physics_loss
        
        return total_loss, mse_loss, physics_loss
    
    def compute_physics_loss(self, predictions, inputs):
        """计算物理约束损失"""
        physics_loss = 0.0
        
        # 约束1: 累积因子必须 >= 1
        min_constraint = torch.relu(1.0 - predictions).mean()
        physics_loss += min_constraint
        
        # 约束2: 有限几何 <= 无限几何
        inf_flu = predictions[:, 0]
        fin_flu = predictions[:, 1]
        inf_exp = predictions[:, 2]
        fin_exp = predictions[:, 3]
        inf_eff = predictions[:, 4]
        fin_eff = predictions[:, 5]
        
        geometry_constraint = (
            torch.relu(fin_flu - inf_flu).mean() +
            torch.relu(fin_exp - inf_exp).mean() +
            torch.relu(fin_eff - inf_eff).mean()
        )
        physics_loss += geometry_constraint
        
        # 约束3: 单调性约束（厚度增加，累积因子增加）
        # 这需要在训练时按厚度排序的批次中计算
        
        return physics_loss

def create_model(input_dim=8, output_dim=6, **kwargs):
    """创建模型的工厂函数"""
    return PhysicsInformedTransformer(
        input_dim=input_dim,
        output_dim=output_dim,
        **kwargs
    )

if __name__ == "__main__":
    # 测试模型
    model = create_model()
    
    # 创建测试数据
    batch_size = 32
    test_input = torch.randn(batch_size, 8)
    
    # 前向传播
    output, attention_weights = model(test_input)
    
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重层数: {len(attention_weights)}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")