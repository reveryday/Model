import numpy as np
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

def load_data(file_path=DATA_PATH, val_size=VAL_SIZE, test_size=TEST_SIZE, 
              random_state=RANDOM_STATE, batch_size=BATCH_SIZE, y_scaler=Y_SCALER):

    data = pd.read_csv(file_path, header=None)  # 不会把文件的第一行当表头，而是自动生成从 0 开始的整数列索引：0, 1, 2, …
    data.columns = [
    "Energy","Shell","MFP",
    "MAC_Total","MAC_Incoherent","MAC_Coherent","MAC_Photoelectric","MAC_Pair_production",
    "Inf_Flu_BUF","Fin_Flu_BUF","Inf_Exp_BUF","Fin_Exp_BUF","Inf_Eff_BUF","Fin_Eff_BUF"]
        
    X = data.drop(columns=["MAC_Coherent"]).iloc[:, :7]   # 去掉 MAC_Coherent 后取前 7 列做特征
    y = data.iloc[:, 8:14].values

    # (1) 对数变换能量
    X["Energy"] = np.log10(X["Energy"]) #范围大约是 [-2, 1]
    E_min, E_max = -2, 1
    X["Energy"] = (X["Energy"] - E_min) / (E_max - E_min)
    
    # (2) 线性归一化厚度 MFP
    X["MFP"] = X["MFP"] / 100.0  # 缩放到 [0,1]
    
    # (3) 对 MAC 类标准化
    scaler_MAC = StandardScaler()
    mac_cols = ["MAC_Total","MAC_Incoherent","MAC_Coherent","MAC_Photoelectric","MAC_Pair_production"]
    X[mac_cols] = scaler_MAC.fit_transform(X[mac_cols])

    X = X.values

    # y_log = np.log(y)  # 直接取 log(y)   
    # y_scaled = y_scaler.fit_transform(y_log)  #对对数变换后的数据进行标准化，提高数值稳定性
    y_scaled = np.log(y)
  
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
        'y_scaler': y_scaler,  # 返回拟合的scaler
    }

# 添加逆变换函数
def inverse_y(predictions, y_scaler):
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    
    # # 首先逆变换StandardScaler
    # y_log_restored = y_scaler.inverse_transform(predictions)    
    # # 逆对数变换：exp(log(y)) = y
    # y_original = np.exp(y_log_restored)
    y_original = np.exp(predictions)
    
    return y_original


def AddNoise(batch, noise_level):
    noise = torch.randn_like(batch) * noise_level * torch.abs(batch).clamp(min=1e-8) 
    noisy_batch = batch + noise  
    noisy_batch = torch.nan_to_num(noisy_batch, nan=0.0, posinf=1e5, neginf=-1e5)
    return noisy_batch


def inverse_transform_predictions(predictions, scaler):
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    return scaler.inverse_transform(predictions)


# 定义Huber Loss类
class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta
    
    def forward(self, input, target):
        abs_error = torch.abs(input - target)
        quadratic = torch.clamp(abs_error, max=self.delta)
        linear = abs_error - quadratic
        loss = 0.5 * quadratic**2 + self.delta * linear
        return torch.mean(loss)
