import pandas as pd

data = pd.read_csv("./data.csv", header=None)
data.columns = [
    "Energy","Shell","MFP",
    "MAC_Total","MAC_Incoherent","MAC_Coherent","MAC_Photoelectric","MAC_Pair_production",
    "Inf_Flu_BUF","Fin_Flu_BUF","Inf_Exp_BUF","Fin_Exp_BUF","Inf_Eff_BUF","Fin_Eff_BUF"]
BUF = data.iloc[:, 8:14].values

count = (BUF >= 1e10).sum()
print("后六列中大于 1e10 的元素个数：", count)