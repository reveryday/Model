import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from conf import *

class GammaDataset(Dataset):
    def __init__(self, features, targets, shell_indices=None):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.shell_indices = shell_indices
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

def analyze_shell_distribution(data):
    """分析Shell分布，为采样权重计算做准备"""
    shell_counts = data['Shell'].value_counts().sort_index()
    print(f"Shell分布统计:")
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
    """
    加载并预处理数据，支持8维输入特征和Shell条件分析
    """
    original_data = pd.read_csv(data_path)
    print(f"原始数据形状: {original_data.shape}")
    
    # 分析Shell分布
    shell_counts = analyze_shell_distribution(original_data)
    
    # --- 数据预处理 ---
    # 1. 连续特征归一化
    continuous_features = ['Energy', 'MFP']
    X_scaler = MinMaxScaler()
    original_data[continuous_features] = X_scaler.fit_transform(original_data[continuous_features])
    
    # 2. MAC系数取log1p后标准化（处理长尾分布）
    mac_features = ['MAC_Total', 'MAC_Incoherent', 'MAC_Coherent', 'MAC_Photoelectric', 'MAC_Pair_production']
    mac_scaler = StandardScaler()
    mac_log = np.log1p(original_data[mac_features].values)
    original_data[mac_features] = mac_scaler.fit_transform(mac_log)
    
    # 3. 目标变量取log1p（保持原有处理）
    buf_columns = TARGET_COLUMNS
    y_scaled = np.log1p(original_data[buf_columns].values)
    
    # --- 提取特征和目标变量 ---
    X = original_data[FEATURE_COLUMNS].values
    shell_indices = original_data['Shell'].values.astype(int)
    
    print(f"扩展后特征形状: {X.shape}")
    print(f"目标变量形状: {y_scaled.shape}")
    print(f"Shell索引范围: {shell_indices.min()} - {shell_indices.max()}")
    
    # --- 数据集划分（考虑Shell分布） ---
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
        'scalers': {
            'X_scaler': X_scaler,
            'mac_scaler': mac_scaler
        },
        'shell_info': {
            'shell_counts': shell_counts,
            'train_shells': shell_train,
            'val_shells': shell_val,
            'test_shells': shell_test
        }
    }

def AddNoise(batch, noise_level=0.01):
    """为输入添加自适应噪声以提高鲁棒性"""
    noise = torch.randn_like(batch) * noise_level * torch.abs(batch).clamp(min=1e-8)
    noisy_batch = batch + noise
    noisy_batch = torch.nan_to_num(noisy_batch, nan=0.0, posinf=1e5, neginf=-1e5)
    return noisy_batch

def inverse_transform_predictions(predictions, use_log1p=True):
    """将预测结果转换回原始尺度"""
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    
    if use_log1p:
        return np.expm1(predictions)  # log1p的逆变换
    return predictions

def get_shell_groups_stats(predictions, targets, shell_indices, shell_groups=SHELL_GROUPS):
    """按Shell分组计算统计指标"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
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
    # 获取所有数据
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