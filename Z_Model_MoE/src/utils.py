import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from conf import *

class GammaDataset(Dataset):
    def __init__(self, features, targets, shell_indices=None):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        if shell_indices is not None:
            self.shell_indices = torch.tensor(shell_indices, dtype=torch.long)
        else:
            self.shell_indices = None
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.shell_indices is not None:
            return self.features[idx], self.targets[idx], self.shell_indices[idx]
        else:
            return self.features[idx], self.targets[idx]

def analyze_shell(data):
    shell_counts = data['Shell'].value_counts().sort_index()
    print("")
    print(f"Shell范围: {shell_counts.index.min()} - {shell_counts.index.max()}")
    print(f"总Shell数: {len(shell_counts)}")
    print(f"样本最多的Shell: {shell_counts.idxmax()} ({shell_counts.max()}个样本)")
    print(f"样本最少的Shell: {shell_counts.idxmin()} ({shell_counts.min()}个样本)")
    return shell_counts

def create_shell_weights(shell_indices):
    """为不平衡的Shell分布创建采样权重"""
    unique_shells, counts = np.unique(shell_indices, return_counts=True)
    # 计算每个shell的权重（反比于频率）
    shell_weights = 1.0 / counts
    # 归一化权重
    shell_weights = shell_weights / shell_weights.sum() * len(shell_weights)
    
    # 为每个样本分配权重
    sample_weights = np.zeros(len(shell_indices))
    for shell, weight in zip(unique_shells, shell_weights):
        mask = shell_indices == shell
        sample_weights[mask] = weight
    
    return sample_weights

def load_data(data_path=DATA_PATH, val_size=VAL_SIZE, test_size=TEST_SIZE, 
              random_state=RANDOM_STATE, batch_size=BATCH_SIZE, use_weighted_sampling=True):
    data = pd.read_csv(data_path)
    
    # 根据FEATURE_COLUMNS选择特征
    feature_cols = FEATURE_COLUMNS.copy()
    if 'MAC_Coherent' in feature_cols:
        feature_cols.remove('MAC_Coherent')  # 移除MAC_Coherent
    
    X = data[feature_cols]
    y = data[TARGET_COLUMNS].values
    
    # 分析Shell number分布
    shell_counts = analyze_shell(data)

    # (1) 能量归一化
    E_min, E_max = 0.01, 10
    X["Energy"] = (X["Energy"] - E_min) / (E_max - E_min)
    
    # (2) 线性归一化厚度 MFP
    X["MFP"] = X["MFP"] / 100.0  # 缩放到 [0,1]
    
    # (3) 对 MAC 类标准化
    scaler_MAC = StandardScaler()
    mac_cols = ["MAC_Total","MAC_Incoherent","MAC_Photoelectric","MAC_Pair_production"]
    X[mac_cols] = scaler_MAC.fit_transform(X[mac_cols])

    X = X.values
    y_scaled = np.log(y)
    shell_indices = data['Shell'].values.astype(int)
    
    # 使用分层划分确保各Shell在训练/验证/测试集中都有代表
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(sss.split(X, shell_indices))
    
    X_train_val, X_test = X[train_val_idx], X[test_idx]
    y_train_val, y_test = y_scaled[train_val_idx], y_scaled[test_idx]
    shell_train_val, shell_test = shell_indices[train_val_idx], shell_indices[test_idx]
    
    # 进一步划分训练和验证集
    adjusted_val_size = val_size / (1 - test_size)
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=adjusted_val_size, random_state=random_state)
    train_idx, val_idx = next(sss_val.split(X_train_val, shell_train_val))
    
    X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
    y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]
    shell_train, shell_val = shell_train_val[train_idx], shell_train_val[val_idx]
    
    # 创建数据集
    train_dataset = GammaDataset(X_train, y_train, shell_train)
    val_dataset = GammaDataset(X_val, y_val, shell_val)
    test_dataset = GammaDataset(X_test, y_test, shell_test)
    
    # 创建数据加载器
    if use_weighted_sampling:
        # 为训练集创建加权采样器以平衡Shell分布
        sample_weights = create_shell_weights(shell_train)
        sampler = WeightedRandomSampler(
            weights=sample_weights, 
            num_samples=len(sample_weights), 
            replacement=True
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'shell_info': {
            'shell_counts': shell_counts,
            'train_shells': shell_train,
            'val_shells': shell_val,
            'test_shells': shell_test
        }
    }

def AddNoise(batch, noise_level=0.01):
    noise = torch.randn_like(batch) * noise_level * torch.abs(batch).clamp(min=1e-8)
    noisy_batch = batch + noise
    noisy_batch = torch.nan_to_num(noisy_batch, nan=0.0, posinf=1e5, neginf=-1e5)
    return noisy_batch

def inverse_transform_predictions(predictions):
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    
    predictions = np.exp(predictions)
    return predictions

def get_shell_groups_stats(predictions, targets, shell_indices, shell_groups=SHELL_GROUPS):
    """按Shell分组计算统计指标"""
    
    stats = {}
    for i, (start_shell, end_shell) in enumerate(shell_groups):
        mask = (shell_indices >= start_shell) & (shell_indices <= end_shell)
        if mask.sum() == 0:
            continue
            
        group_pred = predictions[mask]
        group_true = targets[mask]
        
        stats[f'Shell_{start_shell}-{end_shell}'] = {
            'count': mask.sum(),
            'mse': mean_squared_error(group_true, group_pred),
            'mae': mean_absolute_error(group_true, group_pred),
            'r2': r2_score(group_true, group_pred),
            'mape': np.mean(np.abs((group_true - group_pred) / group_true)) * 100
        }
    
    return stats

def create_thickness_sorted_batches(dataset, batch_size):
    """创建按厚度排序的批次，用于单调性约束"""

    all_features = []
    all_targets = []
    for i in range(len(dataset)):
        feat, targ = dataset[i]
        all_features.append(feat)
        all_targets.append(targ)
    
    features = torch.stack(all_features)
    targets = torch.stack(all_targets)
    
    # 按MFP（厚度）排序
    mfp_values = features[:, 2]  # MFP是第3列
    sorted_indices = torch.argsort(mfp_values)
    
    sorted_features = features[sorted_indices]
    sorted_targets = targets[sorted_indices]
    
    # 创建批次
    batches = []
    for i in range(0, len(sorted_features), batch_size):
        end_idx = min(i + batch_size, len(sorted_features))
        batches.append((sorted_features[i:end_idx], sorted_targets[i:end_idx]))
    
    return batches