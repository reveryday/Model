import pandas as pd

# 读取CSV文件
df = pd.read_csv('BUF_DATA.csv')

# 查询材料104的信息
material_104 = df[df['Material'] == 104]

print('=== 材料104分析 ===')
print(f'总数据行数: {len(material_104)}')
print(f'能量点数量: {material_104["Energy"].nunique()}')
print(f'能量范围: {material_104["Energy"].min()} - {material_104["Energy"].max()} MeV')

print('\n能量值列表:')
energy_values = sorted(material_104['Energy'].unique())
for i, energy in enumerate(energy_values):
    print(f'{i+1:2d}. {energy} MeV')

print(f'\n每个能量点的MFP数据量:')
for energy in energy_values[:5]:  # 显示前5个能量点的详情
    count = len(material_104[material_104['Energy'] == energy])
    print(f'能量 {energy} MeV: {count} 个MFP点')

print('\n材料104的前5行数据预览:')
print(material_104.head())

print('CSV文件统计信息:')
print(f'总行数: {len(df)}')
print(f'材料数量: {df["Material"].nunique()}')
print(f'能量范围: {df["Energy"].min()} - {df["Energy"].max()} MeV')
print(f'MFP范围: {df["MFP"].min()} - {df["MFP"].max()}')

print('\n各材料的数据行数:')
material_counts = df['Material'].value_counts().sort_index()
print(material_counts)

print('\n各材料的能量数据点数:')
for material in sorted(df['Material'].unique()):
    energy_count = df[df['Material'] == material]['Energy'].nunique()
    print(f'材料 {material}: {energy_count} 个能量点')

print('\n前5行数据预览:')
print(df.head())