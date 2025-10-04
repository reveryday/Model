import pandas as pd

def remove_material_column(input_file, output_file):
    """
    从CSV文件中删除Material列（第二列）
    """
    print(f"正在读取文件: {input_file}")
    
    # 读取CSV文件
    df = pd.read_csv(input_file)
    print(f"原始文件形状: {df.shape}")
    print(f"原始列名: {list(df.columns)}")
    
    # 检查是否存在Material列
    if 'Material' not in df.columns:
        print("错误: 文件中没有找到Material列")
        return False
    
    # 删除Material列
    df_new = df.drop('Material', axis=1)
    print(f"删除Material列后的形状: {df_new.shape}")
    print(f"新的列名: {list(df_new.columns)}")
    
    # 保存新文件
    df_new.to_csv(output_file, index=False)
    print(f"已保存到: {output_file}")
    
    # 显示前几行数据进行验证
    print("\n新文件前5行数据:")
    print(df_new.head())
    
    return True

def main():
    # 文件路径
    input_file = 'd:/DL_单层BUF/Model/BUF_DATA_with_MAC.csv'
    output_file = 'd:/DL_单层BUF/Model/BUF_DATA_with_MAC_no_material.csv'
    
    print("开始删除Material列...")
    success = remove_material_column(input_file, output_file)
    
    if success:
        print("\n处理完成！")
        print(f"输入文件: {input_file}")
        print(f"输出文件: {output_file}")
        print("Material列已成功删除")
    else:
        print("处理失败！")

if __name__ == "__main__":
    main()