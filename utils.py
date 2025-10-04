import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
from conf import *

class GammaDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

def load_data(data_path=DATA_PATH, val_size=VAL_SIZE, test_size=TEST_SIZE, 
              random_state=RANDOM_STATE, batch_size=BATCH_SIZE):

    original_data = pd.read_csv(data_path)  # DataFrame是一个类
    print(f"原始数据形状: {original_data.shape}")  # (90168, 10)

    # --- MinMax归一化能量(Energy列)和厚度(MFP列) ---
    X_scaler = MinMaxScaler()
    original_data[['Energy', 'MFP']] = X_scaler.fit_transform(original_data[['Energy', 'MFP']])

    # BUF取log(y+1)
    buf_columns = ['Inf_Flu_BUF', 'Fin_Flu_BUF', 'Inf_Exp_BUF', 'Fin_Exp_BUF', 'Inf_Eff_BUF', 'Fin_Eff_BUF']
    y_scaled = np.log1p(original_data[buf_columns].values)

    # --- 提取特征和目标变量 ---
    feature_columns = ['Energy', 'Shell', 'MFP', 'MAC_Total']
    X = original_data[feature_columns].values
    
    arr = original_data.values   # 转成 numpy
    # 按 (组数=884, 每组102行, 每行10个特征) reshape
    data = arr.reshape(884, 102, 10)
    print(f"reshape后数据形状: {data.shape}")
    print(f"X 形状: {X.shape}")
    print(f"y 形状: {y_scaled.shape}")

    # 划分数据集
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_scaled, test_size=test_size, random_state=random_state, shuffle=True, stratify=None  #每次重新训练时，随机划分的结果都相同        
    )    
    adjusted_val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=adjusted_val_size, random_state=random_state, shuffle=True, stratify=None
    )

    train_dataset = GammaDataset(X_train, y_train)
    val_dataset = GammaDataset(X_val, y_val)
    test_dataset = GammaDataset(X_test, y_test)
    
   
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, pin_memory=True)

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
    }


def AddNoise(batch, noise_level):
    noise = torch.randn_like(batch) * noise_level * torch.abs(batch).clamp(min=1e-8) 
    noisy_batch = batch + noise  
    noisy_batch = torch.nan_to_num(noisy_batch, nan=0.0, posinf=1e5, neginf=-1e5)
    return noisy_batch


def inverse_transform_predictions(predictions, scaler):
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    return scaler.inverse_transform(predictions)
