import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from conf import *


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class MyModel(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, d_model=D_MODEL, 
                 nhead=N_HEAD, num_layers=NUM_LAYERS, dim_feedforward=D_FF, 
                 dropout=DROPOUT, activation=ACTIVATION):
        super(MyModel, self).__init__()
        
        # 增强输入特征投影层
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer编码器
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, #输入和输出特征的维度
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True # 表示输入和输出张量的第一个维度是批次大小
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # 输出预测层
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout/2),  # 减少最后一层的dropout以保留更多信息
            nn.Linear(d_model, output_dim)
        )
        

        # 初始化参数
        self._init_parameters()
    
    def _init_parameters(self):
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
    
    def forward(self, x):
        # x= [batch_size, 102, 4] -> [batch_size, 102, d_model]
        x = self.input_projection(x)
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码器
        x = self.transformer_encoder(x)  # [batch_size, 102, d_model] -> [batch_size, target_seq_len, d_model]
        
        # 取序列的平均值作为特征表示
        x = x.mean(dim=1)  # [batch_size,target_seq_len, d_model] -> [batch_size, d_model]
        
        # 输出
        output = self.output_projection(x)  # [batch_size, d_model] -> [batch_size, output_dim]
        
        return output