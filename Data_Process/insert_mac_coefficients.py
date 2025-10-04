import pandas as pd
import re
from collections import defaultdict

def parse_mac_file(mac_file_path):
    """
    解析MAC文件，创建衰减系数映射表
    返回格式: {(material_id, energy): [MAC_Total, MAC_Incoherent, MAC_Coherent, MAC_Photoelectric, MAC_Pair_production]}
    """
    mac_data = {}
    
    with open(mac_file_path, 'r') as file:
        lines = file.readlines()
    
    current_material = None
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # 查找材料ID行
        if line.startswith('Material_ID'):
            # 解析材料ID
            match = re.search(r'Material_ID\s*=\s*(\d+)', line)
            if match:
                current_material = int(match.group(1))
                print(f"正在处理材料 {current_material}")
                i += 1  # 跳过标题行
                i += 1  # 跳到数据行
                continue
        
        # 如果当前有材料ID，解析能量和衰减系数数据
        if current_material is not None and line and not line.startswith('Energy'):
            try:
                # 分割数据行
                parts = line.split()
                if len(parts) >= 6:  # 确保有足够的列
                    energy = float(parts[0])
                    mac_total = float(parts[1])
                    mac_incoherent = float(parts[2])
                    mac_coherent = float(parts[3])
                    mac_photoelectric = float(parts[4])
                    mac_pair_production = float(parts[5])
                    
                    # 存储到映射表
                    mac_data[(current_material, energy)] = [
                        mac_total, mac_incoherent, mac_coherent, 
                        mac_photoelectric, mac_pair_production
                    ]
            except (ValueError, IndexError) as e:
                print(f"解析行时出错: {line.strip()}, 错误: {e}")
        
        i += 1
    
    print(f"成功解析 {len(mac_data)} 条衰减系数记录")
    return mac_data

def insert_mac_coefficients(csv_file_path, mac_data, output_file_path):
    """
    读取BUF_DATA.csv文件，插入衰减系数，并保存到新文件
    """
    print("正在读取BUF_DATA.csv文件...")
    df = pd.read_csv(csv_file_path)
    
    print(f"原始数据形状: {df.shape}")
    print(f"原始列名: {list(df.columns)}")
    
    # 创建新的衰减系数列
    mac_columns = ['MAC_Total', 'MAC_Incoherent', 'MAC_Coherent', 'MAC_Photoelectric', 'MAC_Pair_production']
    
    # 初始化新列
    for col in mac_columns:
        df[col] = 0.0
    
    # 统计匹配和未匹配的记录
    matched_count = 0
    unmatched_count = 0
    unmatched_materials = set()
    unmatched_energies = set()
    
    print("正在匹配衰减系数...")
    
    # 遍历每一行，查找对应的衰减系数
    for idx, row in df.iterrows():
        material = int(row['Material'])
        energy = float(row['Energy'])
        
        # 查找匹配的衰减系数
        key = (material, energy)
        if key in mac_data:
            # 找到匹配的衰减系数
            coefficients = mac_data[key]
            df.loc[idx, 'MAC_Total'] = coefficients[0]
            df.loc[idx, 'MAC_Incoherent'] = coefficients[1]
            df.loc[idx, 'MAC_Coherent'] = coefficients[2]
            df.loc[idx, 'MAC_Photoelectric'] = coefficients[3]
            df.loc[idx, 'MAC_Pair_production'] = coefficients[4]
            matched_count += 1
        else:
            unmatched_count += 1
            unmatched_materials.add(material)
            unmatched_energies.add(energy)
    
    print(f"匹配成功: {matched_count} 条记录")
    print(f"未匹配: {unmatched_count} 条记录")
    
    if unmatched_count > 0:
        print(f"未匹配的材料: {sorted(unmatched_materials)}")
        print(f"未匹配的能量点数量: {len(unmatched_energies)}")
        print(f"部分未匹配的能量: {sorted(list(unmatched_energies))[:10]}...")
    
    # 重新排列列的顺序：在MFP后插入五个衰减系数列
    original_columns = ['Energy', 'Material', 'Shell', 'MFP']
    mac_columns = ['MAC_Total', 'MAC_Incoherent', 'MAC_Coherent', 'MAC_Photoelectric', 'MAC_Pair_production']
    remaining_columns = ['Inf_Flu_BUF', 'Fin_Flu_BUF', 'Inf_Exp_BUF', 'Fin_Exp_BUF', 'Inf_Eff_BUF', 'Fin_Eff_BUF']
    
    new_column_order = original_columns + mac_columns + remaining_columns
    df = df[new_column_order]
    
    print(f"新数据形状: {df.shape}")
    print(f"新列名: {list(df.columns)}")
    
    # 保存到新文件
    print(f"正在保存到 {output_file_path}...")
    df.to_csv(output_file_path, index=False)
    print("保存完成!")
    
    return df

def main():
    # 文件路径
    mac_file = r'd:\DL_单层BUF\Model\MAC'
    csv_file = r'd:\DL_单层BUF\Model\BUF_DATA.csv'
    output_file = r'd:\DL_单层BUF\Model\BUF_DATA_with_MAC.csv'
    
    print("开始处理衰减系数插入...")
    
    # 解析MAC文件
    print("步骤1: 解析MAC文件...")
    mac_data = parse_mac_file(mac_file)
    
    # 插入衰减系数
    print("步骤2: 插入衰减系数到BUF_DATA.csv...")
    df_result = insert_mac_coefficients(csv_file, mac_data, output_file)
    
    # 显示结果预览
    print("\n结果预览:")
    print(df_result.head(10))
    
    print(f"\n处理完成! 结果已保存到: {output_file}")

if __name__ == "__main__":
    main()