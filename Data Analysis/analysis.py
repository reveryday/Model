import pandas as pd
import numpy as np

# 读取数据
data = pd.read_csv("./data.csv", header=None)
data.columns = [
    "Energy","Shell","MFP",
    "MAC_Total","MAC_Incoherent","MAC_Coherent","MAC_Photoelectric","MAC_Pair_production",
    "Inf_Flu_BUF","Fin_Flu_BUF","Inf_Exp_BUF","Fin_Exp_BUF","Inf_Eff_BUF","Fin_Eff_BUF"
]

BUF = data.iloc[:, 8:14].values

mask = (BUF > 1e25).any(axis=1)
filtered_data = data[mask]
count = (BUF >= 1e25).sum()
print("BUF中大于 1e25 的元素总个数：", count)
print("包含这些大数的行数：", mask.sum())

MAC = data.iloc[:, 3].values
count = (MAC >= 5).sum()
print("MAC中大于 5 的元素总个数：", count)

# 导出为CSV文件
# filtered_data.to_csv("BUF_over_1e25.csv", index=False, encoding="utf-8")

# print("Successfully exported BUF_over_1e25.csv")