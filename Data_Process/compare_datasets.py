import pandas as pd

def compare_datasets():
    """对比原始数据集和处理后的数据集"""
    
    print("=== 数据集对比分析 ===\n")
    
    # 读取原始数据集
    df_original = pd.read_csv('dataset.csv')
    print("原始数据集:")
    print(f"  形状: {df_original.shape}")
    print(f"  列数: {len(df_original.columns)}")
    print(f"  列名: {list(df_original.columns)}")
    
    # 读取处理后的数据集
    df_reduced = pd.read_csv('dataset_reduced.csv')
    print("\n处理后数据集:")
    print(f"  形状: {df_reduced.shape}")
    print(f"  列数: {len(df_reduced.columns)}")
    print(f"  列名: {list(df_reduced.columns)}")
    
    # 显示被删除的列
    removed_columns = set(df_original.columns) - set(df_reduced.columns)
    print(f"\n被删除的列: {list(removed_columns)}")
    
    # 验证数据一致性（检查保留的列是否相同）
    print("\n=== 数据一致性验证 ===")
    
    # 检查前几行数据
    print("\n原始数据集前3行（保留的列）:")
    cols_to_check = ['Energy', 'Shell', 'MFP', 'MAC_Total']
    print(df_original[cols_to_check].head(3))
    
    print("\n处理后数据集前3行（对应列）:")
    print(df_reduced[cols_to_check].head(3))
    
    # 检查累积因子列
    print("\n原始数据集累积因子列前3行:")
    buf_cols = ['Inf_Flu_BUF', 'Fin_Flu_BUF', 'Inf_Exp_BUF', 'Fin_Exp_BUF', 'Inf_Eff_BUF', 'Fin_Eff_BUF']
    print(df_original[buf_cols].head(3))
    
    print("\n处理后数据集累积因子列前3行:")
    print(df_reduced[buf_cols].head(3))
    
    # 验证数据完全一致
    is_consistent = True
    for col in df_reduced.columns:
        if not df_original[col].equals(df_reduced[col]):
            print(f"警告: 列 {col} 数据不一致!")
            is_consistent = False
    
    if is_consistent:
        print("\n✅ 数据一致性验证通过！保留的列数据完全一致。")
    else:
        print("\n❌ 数据一致性验证失败！")
    
    print(f"\n=== 总结 ===")
    print(f"✅ 成功删除了4列: {list(removed_columns)}")
    print(f"✅ 数据集从 {df_original.shape[1]} 列减少到 {df_reduced.shape[1]} 列")
    print(f"✅ 保留了 {df_reduced.shape[0]} 行数据")
    print(f"✅ 新数据集保存为: dataset_reduced.csv")

if __name__ == "__main__":
    compare_datasets()