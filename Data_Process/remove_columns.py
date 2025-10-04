import pandas as pd

def remove_columns_5_to_8():
    """去掉数据集中的第5-8列（MAC_Incoherent, MAC_Coherent, MAC_Photoelectric, MAC_Pair_production）"""
    
    print("正在读取原始数据集...")
    # 读取原始数据集
    df = pd.read_csv('dataset.csv')
    
    print(f"原始数据集形状: {df.shape}")
    print(f"原始列名: {list(df.columns)}")
    
    # 获取要保留的列（去掉第5-8列，即索引4-7）
    columns_to_keep = list(df.columns)
    columns_to_remove = columns_to_keep[4:8]  # 第5-8列（索引4-7）
    
    print(f"要删除的列: {columns_to_remove}")
    
    # 删除指定列
    df_new = df.drop(columns=columns_to_remove)
    
    print(f"新数据集形状: {df_new.shape}")
    print(f"新列名: {list(df_new.columns)}")
    
    # 保存新数据集
    output_filename = 'dataset_reduced.csv'
    df_new.to_csv(output_filename, index=False)
    
    print(f"新数据集已保存为: {output_filename}")
    
    # 显示前几行数据进行验证
    print("\n新数据集前5行:")
    print(df_new.head())
    
    return df_new

if __name__ == "__main__":
    remove_columns_5_to_8()