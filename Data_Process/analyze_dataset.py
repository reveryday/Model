import pandas as pd
import numpy as np

# 读取数据集
df = pd.read_csv('dataset.csv')

print(f'数据集形状: {df.shape}')
print(f'能量范围: {df["Energy"].min()} - {df["Energy"].max()}')
print(f'厚度范围: {df["MFP"].min()} - {df["MFP"].max()}')
print(f'壳层数: {df["Shell"].nunique()}')
print(f'唯一能量值数量: {df["Energy"].nunique()}')

print('\n输入特征统计 (前8列):')
print(df.iloc[:, :8].describe())

print('\n输出特征统计 (后6列):')
print(df.iloc[:, 8:].describe())

print('\n数据类型:')
print(df.dtypes)

print('\n缺失值检查:')
print(df.isnull().sum())