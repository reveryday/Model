import os
import csv
import re

FILE_PATH = "./BUF_DATA"
def parse_buf_data(file_path=FILE_PATH):

    data_rows = []    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 按材料分割数据
    material_blocks = re.split(r'Material Number =', content)[1:]  # 跳过第一个空块
    
    for block in material_blocks:
        lines = block.strip().split('\n')
        if len(lines) < 8:  # 确保有足够的行数
            continue
            
        # 解析材料编号和壳层数据
        first_line = lines[0].strip()
        parts = first_line.split()
        material_number = int(parts[0])
        # 查找Shell Number的位置
        shell_idx = -1
        for i, part in enumerate(parts):
            if part == "Shell" and i+2 < len(parts) and parts[i+1] == "Number":
                shell_idx = i + 3  # "Shell Number = X"
                break
        shell_number = int(parts[shell_idx]) if shell_idx != -1 and shell_idx < len(parts) else 0
        
        # 查找所有能量块
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # 查找能量行
            if line.startswith("Energy ="):
                energy_str = line.split('=')[1].strip().split()[0]
                energy = float(energy_str)
                
                # 查找MFP行
                i += 1
                while i < len(lines) and not lines[i].strip().startswith("MFP"):
                    i += 1
                
                if i >= len(lines):
                    break
                    
                # 解析MFP值
                mfp_line = lines[i].strip()
                mfp_values = mfp_line.split()[1:]  # 跳过"MFP"标签
                
                # 读取六种累积因子
                factor_names = ["Inf_Flu_BUF", "Fin_Flu_BUF", "Inf_Exp_BUF", "Fin_Exp_BUF", "Inf_Eff_BUF", "Fin_Eff_BUF"]
                factor_values = {}
                
                for j in range(6):
                    i += 1
                    if i < len(lines):
                        factor_line = lines[i].strip()
                        if factor_line:  # 确保行不为空
                            parts = factor_line.split()
                            if len(parts) > 1:  # 确保有数据
                                factor_name = parts[0]
                                factor_values[factor_name] = parts[1:]
                
                # 为每个MFP值创建一行数据
                for idx, mfp in enumerate(mfp_values):
                    try:
                        row = {
                            "Energy": energy,
                            "Material": material_number,
                            "Shell": shell_number,
                            "MFP": float(mfp)
                        }
                        
                        # 添加六种累积因子
                        for factor_name in factor_names:
                            if factor_name in factor_values and idx < len(factor_values[factor_name]):
                                try:
                                    row[factor_name] = float(factor_values[factor_name][idx])
                                except ValueError:
                                    row[factor_name] = factor_values[factor_name][idx]
                            else:
                                row[factor_name] = ""
                        
                        data_rows.append(row)
                    except ValueError:
                        continue  # 跳过无法解析的行
            
            i += 1
    
    return data_rows

def write_to_csv(data, output_file):
    """
    将解析的数据写入CSV文件
    """
    if not data:
        print("没有数据可写入")
        return
    
    fieldnames = ["Energy", "Material", "Shell", "MFP", "Inf_Flu_BUF", "Fin_Flu_BUF", 
                 "Inf_Exp_BUF", "Fin_Exp_BUF", "Inf_Eff_BUF", "Fin_Eff_BUF"]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    print(f"数据已成功写入 {output_file}")
    print(f"总共处理了 {len(data)} 行数据")

def main():
    input_file = "d:\\DL_单层BUF\\Model\\BUF_DATA"
    output_file = "d:\\DL_单层BUF\\Model\\BUF_DATA.csv"
    
    print("开始解析BUF_DATA文件...")
    data = parse_buf_data(input_file)
    print(f"解析完成，共找到 {len(data)} 行数据")
    
    write_to_csv(data, output_file)

if __name__ == "__main__":
    main()