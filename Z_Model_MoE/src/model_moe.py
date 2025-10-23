import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from conf import *

class PositionalEncoding(nn.Module):
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

class ShellEmbedding(nn.Module):
    """Shell条件嵌入层"""
    def __init__(self, num_shells, embedding_dim):
        super(ShellEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_shells, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, shell_indices):
        # shell_indices: [batch_size]
        embedded = self.embedding(shell_indices)  # [batch_size, embedding_dim]
        return self.layer_norm(embedded)

class ExpertNetwork(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super(ExpertNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x):
        return self.network(x)

class ShellGatingNetwork(nn.Module):
    """Shell条件门控网络"""
    def __init__(self, input_dim, shell_embedding_dim, num_experts, hidden_dim=128):
        super(ShellGatingNetwork, self).__init__()
        self.num_experts = num_experts
        
        # 结合输入特征和Shell嵌入
        combined_dim = input_dim + shell_embedding_dim
        
        self.gate_network = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_experts)
        )
        
    def forward(self, x, shell_embedding):
        # x: [batch_size, input_dim]
        # shell_embedding: [batch_size, shell_embedding_dim]
        combined = torch.cat([x, shell_embedding], dim=-1)
        gate_logits = self.gate_network(combined)
        gate_weights = F.softmax(gate_logits, dim=-1)
        return gate_weights

class MixtureOfExperts(nn.Module):
    """混合专家层"""
    def __init__(self, input_dim, shell_embedding_dim, num_experts, expert_hidden_dim, output_dim):
        super(MixtureOfExperts, self).__init__()
        self.num_experts = num_experts
        
        # 创建专家网络
        self.experts = nn.ModuleList([
            ExpertNetwork(input_dim, expert_hidden_dim, output_dim)
            for _ in range(num_experts)
        ])
        
        # 门控网络
        self.gating = ShellGatingNetwork(input_dim, shell_embedding_dim, num_experts)
        
    def forward(self, x, shell_embedding):
        # 获取门控权重
        gate_weights = self.gating(x, shell_embedding)  # [batch_size, num_experts]
        
        # 计算所有专家的输出
        expert_outputs = []
        for expert in self.experts:
            expert_output = expert(x)  # [batch_size, output_dim]
            expert_outputs.append(expert_output)
        
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch_size, num_experts, output_dim]
        
        # 加权组合专家输出
        gate_weights = gate_weights.unsqueeze(-1)  # [batch_size, num_experts, 1]
        moe_output = torch.sum(gate_weights * expert_outputs, dim=1)  # [batch_size, output_dim]
        
        return moe_output, gate_weights.squeeze(-1)

class SharedBackbone(nn.Module):
    """共享主干网络（基于Transformer）"""
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super(SharedBackbone, self).__init__()
        self.d_model = d_model
        
        # 输入投影
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出投影
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # x: [batch_size, input_dim]
        batch_size = x.size(0)
        
        # 输入投影
        x = self.input_projection(x)  # [batch_size, d_model]
        
        # 添加序列维度用于Transformer
        x = x.unsqueeze(1)  # [batch_size, 1, d_model]
        
        # 位置编码
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)  # [batch_size, 1, d_model]
        
        # Transformer编码
        x = self.transformer_encoder(x)  # [batch_size, 1, d_model]
        
        # 移除序列维度
        x = x.squeeze(1)  # [batch_size, d_model]
        
        # 输出投影
        x = self.output_projection(x)  # [batch_size, d_model//2]
        
        return x

class MultiTaskHead(nn.Module):
    """多任务输出头"""
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(MultiTaskHead, self).__init__()
        self.output_dim = output_dim
        
        # 为每个输出任务创建独立的头
        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, 1)
            ) for _ in range(output_dim)
        ])
        
    def forward(self, x):
        outputs = []
        for head in self.task_heads:
            output = head(x)
            outputs.append(output)
        return torch.cat(outputs, dim=-1)

class ShellMoEModel(nn.Module):
    def __init__(self, 
                 input_dim=INPUT_DIM,
                 output_dim=OUTPUT_DIM,
                 num_shells=NUM_SHELLS,
                 shell_embedding_dim=EMBED_DIM,
                 num_experts=NUM_EXPERTS,
                 expert_hidden_dim=EXPERT_HIDDEN_DIM,
                 backbone_d_model=BACKBONE_D_MODEL,
                 backbone_nhead=BACKBONE_NHEAD,
                 backbone_layers=BACKBONE_LAYERS,
                 backbone_dim_feedforward=BACKBONE_DIM_FEEDFORWARD,
                 dropout=0.1):
        super(ShellMoEModel, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_shells = num_shells
        self.num_experts = num_experts
        
        # Shell嵌入层
        self.shell_embedding = ShellEmbedding(num_shells, shell_embedding_dim)
        
        # 共享主干网络
        self.shared_backbone = SharedBackbone(
            input_dim=input_dim,
            d_model=backbone_d_model,
            nhead=backbone_nhead,
            num_layers=backbone_layers,
            dim_feedforward=backbone_dim_feedforward,
            dropout=dropout
        )
        
        # 混合专家层
        backbone_output_dim = backbone_d_model // 2
        self.moe_layer = MixtureOfExperts(
            input_dim=backbone_output_dim,
            shell_embedding_dim=shell_embedding_dim,
            num_experts=num_experts,
            expert_hidden_dim=expert_hidden_dim,
            output_dim=expert_hidden_dim
        )
        
        # 多任务输出头
        self.output_head = MultiTaskHead(
            input_dim=expert_hidden_dim,
            output_dim=output_dim,
            hidden_dim=expert_hidden_dim // 2
        )
        
        # 初始化权重
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.1)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, x, shell_indices=None):
        """
        前向传播
        Args:
            x: [batch_size, input_dim] 输入特征
            shell_indices: [batch_size] Shell索引，如果为None则从输入中提取
        """
        batch_size = x.size(0)
        
        # 如果没有提供shell_indices，从输入特征中提取
        if shell_indices is None:
            # 假设Shell是输入特征的第2列（索引1）
            shell_indices = x[:, 1].long()
        
        # 确保shell_indices在有效范围内
        shell_indices = torch.clamp(shell_indices, 0, self.num_shells - 1)
        
        # Shell嵌入
        shell_emb = self.shell_embedding(shell_indices)  # [batch_size, shell_embedding_dim]
        
        # 共享主干网络
        backbone_output = self.shared_backbone(x)  # [batch_size, backbone_output_dim]
        
        # 混合专家层
        moe_output, gate_weights = self.moe_layer(backbone_output, shell_emb)
        
        # 多任务输出头
        final_output = self.output_head(moe_output)  # [batch_size, output_dim]
        
        return {
            'predictions': final_output,
            'gate_weights': gate_weights,
            'shell_embedding': shell_emb,
            'backbone_features': backbone_output
        }
    
    def get_expert_usage(self, gate_weights):
        """计算专家使用统计"""
        # gate_weights: [batch_size, num_experts]
        expert_usage = torch.mean(gate_weights, dim=0)  # [num_experts]
        return expert_usage
    
    def compute_load_balancing_loss(self, gate_weights, alpha=0.01):
        """计算负载均衡损失"""
        # 鼓励专家使用的均匀分布
        expert_usage = self.get_expert_usage(gate_weights)
        target_usage = torch.ones_like(expert_usage) / self.num_experts
        load_balancing_loss = F.mse_loss(expert_usage, target_usage)
        return alpha * load_balancing_loss

def create_model():
    """创建模型实例"""
    model = ShellMoEModel()
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"模型总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"模型结构:")
    print(f"  - 输入维度: {INPUT_DIM}")
    print(f"  - 输出维度: {OUTPUT_DIM}")
    print(f"  - Shell数量: {NUM_SHELLS}")
    print(f"  - 专家数量: {NUM_EXPERTS}")
    print(f"  - Shell嵌入维度: {SHELL_EMBEDDING_DIM}")
    print(f"  - 主干网络维度: {BACKBONE_D_MODEL}")
    
    return model

if __name__ == "__main__":
    # 测试模型
    model = create_model()
    
    # 创建测试数据
    batch_size = 32
    test_input = torch.randn(batch_size, INPUT_DIM)
    test_shell_indices = torch.randint(0, NUM_SHELLS, (batch_size,))
    
    # 前向传播
    with torch.no_grad():
        output = model(test_input, test_shell_indices)
        
    print(f"\n测试结果:")
    print(f"输入形状: {test_input.shape}")
    print(f"Shell索引形状: {test_shell_indices.shape}")
    print(f"预测输出形状: {output['predictions'].shape}")
    print(f"门控权重形状: {output['gate_weights'].shape}")
    print(f"专家使用分布: {model.get_expert_usage(output['gate_weights'])}")