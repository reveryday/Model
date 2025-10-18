import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from conf import *


class PositionalEncoding(nn.Module):
    # max_len: Transformer模型支持的最大序列长度
    def __init__(self, d_model, max_len=102):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 取出所有偶数序列转为浮点型，
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # 在最前面加一个 batch 维度
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
        # [batch_size, input_dim(8)] -> [batch_size, 1, input_dim]
        x = x.unsqueeze(1)
        # [batch_size, 1, input_dim] -> [batch_size, 3, input_dim]
        x = x.repeat(1, 4, 1)
        #  [batch_size, 3, input_dim] -> [batch_size, 3, d_model]
        x = self.input_projection(x)
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码器
        x = self.transformer_encoder(x)  # [batch_size, 3, d_model] -> [batch_size, target_seq_len, d_model]
        
        # 取序列的平均值作为输出
        x = x.mean(dim=1)  # [batch_size,target_seq_len, d_model] -> [batch_size, d_model]
        
        # 输出
        output = self.output_projection(x)  # [batch_size, d_model] -> [batch_size, output_dim]
        
        return output

class MLPModel(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM,
                 dropout=DROPOUT, activation=ACTIVATION):
        
        super(MLPModel, self).__init__()
        act = nn.GELU() if activation.lower() == "gelu" else nn.ReLU()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LayerNorm(32),
            act,

            nn.Linear(32, 128),
            nn.LayerNorm(128),
            act,
            nn.Dropout(dropout),

            nn.Linear(128, 256),
            nn.LayerNorm(256),
            act,
            nn.Dropout(dropout),

            nn.Linear(256, 64),
            nn.LayerNorm(64),
            act,
            nn.Dropout(dropout),

            nn.Linear(64, output_dim)
        )
        self._init_parameters()

    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # [batch_size, INPUT_DIM] -> [batch_size, OUTPUT_DIM]
        return self.net(x)