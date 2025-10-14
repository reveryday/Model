#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV文件数值范围分析脚本
分析BUF_DATA_with_MAC_no_material.csv文件中每一列的数值范围和统计信息
"""

import pandas as pd
import numpy as np
import os

def analyze_csv_ranges(csv_file_path):
    """
    分析CSV文件每一列的数值范围和统计信息
    
    Args:
        csv_file_path (str): CSV文件路径
    """
    print(f"正在分析文件: {csv_file_path}")
    print("=" * 80)
    
    # 读取CSV文件
    try:
        df = pd.read_csv(csv_file_path)
        print(f"文件读取成功！")
        print(f"数据形状: {df.shape[0]} 行 × {df.shape[1]} 列")
        print(f"列名: {list(df.columns)}")
        print("=" * 80)
        
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return
    
    # 分析每一列的统计信息
    print("各列数值范围和统计信息:")
    print("=" * 80)
    
    for col in df.columns:
        print(f"\n列名: {col}")
        print("-" * 50)
        
        # 基本统计信息
        col_data = df[col]
        
        # 检查数据类型
        print(f"数据类型: {col_data.dtype}")
        print(f"非空值数量: {col_data.count()}")
        print(f"空值数量: {col_data.isnull().sum()}")
        
        # 如果是数值型数据，计算统计信息
        if pd.api.types.is_numeric_dtype(col_data):
            print(f"最小值: {col_data.min():.6f}")
            print(f"最大值: {col_data.max():.6f}")
            print(f"平均值: {col_data.mean():.6f}")
            print(f"中位数: {col_data.median():.6f}")
            print(f"标准差: {col_data.std():.6f}")
            print(f"方差: {col_data.var():.6f}")
            
            # 分位数信息
            print(f"25%分位数: {col_data.quantile(0.25):.6f}")
            print(f"75%分位数: {col_data.quantile(0.75):.6f}")
            
            # 唯一值信息
            unique_count = col_data.nunique()
            print(f"唯一值数量: {unique_count}")
            
            # 如果唯一值较少，显示所有唯一值
            if unique_count <= 20:
                unique_values = sorted(col_data.unique())
                print(f"唯一值: {unique_values}")
            else:
                # 显示前10个和后10个唯一值
                unique_values = sorted(col_data.unique())
                print(f"唯一值(前10个): {unique_values[:10]}")
                print(f"唯一值(后10个): {unique_values[-10:]}")
                
        else:
            # 非数值型数据
            unique_values = col_data.unique()
            print(f"唯一值: {unique_values}")
    
    # 生成汇总表
    print("\n" + "=" * 80)
    print("数值列汇总表:")
    print("=" * 80)
    
    # 创建汇总DataFrame
    summary_data = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            col_data = df[col]
            summary_data.append({
                '列名': col,
                '最小值': f"{col_data.min():.6f}",
                '最大值': f"{col_data.max():.6f}",
                '平均值': f"{col_data.mean():.6f}",
                '标准差': f"{col_data.std():.6f}",
                '唯一值数量': col_data.nunique()
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
    
    # 保存汇总结果到文件
    output_file = os.path.join(os.path.dirname(csv_file_path), 'column_analysis_summary.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"CSV文件分析报告\n")
        f.write(f"文件: {csv_file_path}\n")
        f.write(f"数据形状: {df.shape[0]} 行 × {df.shape[1]} 列\n")
        f.write(f"列名: {list(df.columns)}\n\n")
        
        for col in df.columns:
            f.write(f"列名: {col}\n")
            f.write("-" * 50 + "\n")
            
            col_data = df[col]
            f.write(f"数据类型: {col_data.dtype}\n")
            f.write(f"非空值数量: {col_data.count()}\n")
            f.write(f"空值数量: {col_data.isnull().sum()}\n")
            
            if pd.api.types.is_numeric_dtype(col_data):
                f.write(f"最小值: {col_data.min():.6f}\n")
                f.write(f"最大值: {col_data.max():.6f}\n")
                f.write(f"平均值: {col_data.mean():.6f}\n")
                f.write(f"中位数: {col_data.median():.6f}\n")
                f.write(f"标准差: {col_data.std():.6f}\n")
                f.write(f"25%分位数: {col_data.quantile(0.25):.6f}\n")
                f.write(f"75%分位数: {col_data.quantile(0.75):.6f}\n")
                f.write(f"唯一值数量: {col_data.nunique()}\n")
            f.write("\n")
    
    print(f"\n分析结果已保存到: {output_file}")

if __name__ == "__main__":
    # CSV文件路径
    csv_file = r"d:\DL_单层BUF\Model\Data_Process\BUF_DATA_with_MAC_no_material.csv"
    
    # 检查文件是否存在
    if os.path.exists(csv_file):
        analyze_csv_ranges(csv_file)
    else:
        print(f"文件不存在: {csv_file}")