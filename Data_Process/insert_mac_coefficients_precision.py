import pandas as pd
import numpy as np

def parse_mac_file(mac_file_path):
    """
    解析MAC文件，提取衰减系数数据
    """
    mac_data = {}
    
    with open(mac_file_path, 'r') as file:
        lines = file.readlines()
    
    current_material = None
    current_shell = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if 'Material_ID' in line and 'Shell_Number' in line:
            # 解析材料ID和壳层号
            parts = line.split()
            # 格式: Material_ID = 4 Shell_Number = 0
            material_idx = parts.index('=') + 1
            material_id = int(parts[material_idx])
            shell_idx = parts.index('=', material_idx + 1) + 1
            shell_number = int(parts[shell_idx])
            current_material = material_id
            current_shell = shell_number
            continue
        
        # 跳过表头行
        if 'Energy' in line and 'MAC_Total' in line:
            continue
            
        # 解析数据行
        if current_material is not None and current_shell is not None:
            try:
                parts = line.split()
                if len(parts) == 6:  # Energy + 5 MAC values
                    energy = float(parts[0])
                    mac_total = parts[1]  # 保持原始字符串格式
                    mac_incoherent = parts[2]
                    mac_coherent = parts[3]
                    mac_photoelectric = parts[4]
                    mac_pair_production = parts[5]
                    
                    key = (current_material, current_shell, energy)
                    mac_data[key] = {
                        'MAC_Total': mac_total,
                        'MAC_Incoherent': mac_incoherent,
                        'MAC_Coherent': mac_coherent,
                        'MAC_Photoelectric': mac_photoelectric,
                        'MAC_Pair_production': mac_pair_production
                    }
            except (ValueError, IndexError):
                continue
    
    return mac_data

def format_scientific_notation(value, precision=4):
    """
    将数值格式化为科学计数法，保持与BUF_DATA文件一致的格式
    """
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return value
    
    if value == 0:
        return f"{0:.{precision}f}E+00"
    
    # 格式化为科学计数法
    formatted = f"{value:.{precision}E}"
    
    # 确保指数部分有正确的格式 (E+00 或 E-00)
    if 'E' in formatted:
        mantissa, exponent = formatted.split('E')
        exp_num = int(exponent)
        if exp_num >= 0:
            formatted = f"{mantissa}E+{exp_num:02d}"
        else:
            formatted = f"{mantissa}E{exp_num:03d}"
    
    return formatted

def insert_mac_coefficients_with_precision(buf_data_path, mac_data, output_path):
    """
    将MAC系数插入到BUF_DATA中，保持原始精度格式
    """
    # 读取BUF_DATA文件
    df = pd.read_csv(buf_data_path)
    print(f"读取BUF_DATA文件，原始形状: {df.shape}")
    
    # 创建新的MAC列
    mac_columns = ['MAC_Total', 'MAC_Incoherent', 'MAC_Coherent', 'MAC_Photoelectric', 'MAC_Pair_production']
    for col in mac_columns:
        df[col] = ''
    
    # 重新排列列的顺序，将MAC列插入到MFP和Inf_Flu_BUF之间
    original_columns = ['Energy', 'Material', 'Shell', 'MFP']
    mac_columns_list = ['MAC_Total', 'MAC_Incoherent', 'MAC_Coherent', 'MAC_Photoelectric', 'MAC_Pair_production']
    remaining_columns = ['Inf_Flu_BUF', 'Fin_Flu_BUF', 'Inf_Exp_BUF', 'Fin_Exp_BUF', 'Inf_Eff_BUF', 'Fin_Eff_BUF']
    
    new_column_order = original_columns + mac_columns_list + remaining_columns
    df = df[new_column_order]
    
    matched_count = 0
    unmatched_count = 0
    
    # 为每一行添加MAC系数
    for idx, row in df.iterrows():
        material = int(row['Material'])
        shell = int(row['Shell'])
        energy = float(row['Energy'])
        
        key = (material, shell, energy)
        
        if key in mac_data:
            # 直接使用原始字符串格式的MAC数据
            df.at[idx, 'MAC_Total'] = mac_data[key]['MAC_Total']
            df.at[idx, 'MAC_Incoherent'] = mac_data[key]['MAC_Incoherent']
            df.at[idx, 'MAC_Coherent'] = mac_data[key]['MAC_Coherent']
            df.at[idx, 'MAC_Photoelectric'] = mac_data[key]['MAC_Photoelectric']
            df.at[idx, 'MAC_Pair_production'] = mac_data[key]['MAC_Pair_production']
            matched_count += 1
        else:
            # 如果没有匹配，设置为0并格式化
            df.at[idx, 'MAC_Total'] = "0.0000000000E+00"
            df.at[idx, 'MAC_Incoherent'] = "0.0000000000E+00"
            df.at[idx, 'MAC_Coherent'] = "0.0000000000E+00"
            df.at[idx, 'MAC_Photoelectric'] = "0.0000000000E+00"
            df.at[idx, 'MAC_Pair_production'] = "0.0000000000E+00"
            unmatched_count += 1
    
    print(f"匹配的记录数: {matched_count}")
    print(f"未匹配的记录数: {unmatched_count}")
    
    # 格式化其他数值列以保持一致性
    # Energy列保持原始格式
    df['Energy'] = df['Energy'].apply(lambda x: format_scientific_notation(x, 4))
    
    # MFP列格式化
    df['MFP'] = df['MFP'].apply(lambda x: format_scientific_notation(x, 1) if x != 0 else "0.0")
    
    # 累积因子列格式化
    cumulative_columns = ['Inf_Flu_BUF', 'Fin_Flu_BUF', 'Inf_Exp_BUF', 'Fin_Exp_BUF', 'Inf_Eff_BUF', 'Fin_Eff_BUF']
    for col in cumulative_columns:
        df[col] = df[col].apply(lambda x: format_scientific_notation(x, 4))
    
    # 保存到新文件
    df.to_csv(output_path, index=False)
    print(f"已保存到: {output_path}")
    print(f"新文件形状: {df.shape}")
    
    return df

def main():
    # 文件路径
    mac_file_path = 'd:/DL_单层BUF/Model/MAC'
    buf_data_path = 'd:/DL_单层BUF/Model/BUF_DATA.csv'
    output_path = 'd:/DL_单层BUF/Model/BUF_DATA_with_MAC.csv'
    
    print("开始解析MAC文件...")
    mac_data = parse_mac_file(mac_file_path)
    print(f"解析完成，共提取 {len(mac_data)} 条衰减系数记录")
    
    print("\n开始插入MAC系数到BUF_DATA...")
    result_df = insert_mac_coefficients_with_precision(buf_data_path, mac_data, output_path)
    
    print("\n处理完成！")
    print(f"输出文件: {output_path}")
    print(f"文件包含 {result_df.shape[0]} 行数据，{result_df.shape[1]} 列")

if __name__ == "__main__":
    main()