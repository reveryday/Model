import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
matplotlib.rcParams['axes.unicode_minus'] = False    # 正常显示负号


df = pd.read_csv('data.csv', header=None)
shell_col = df.columns[1]
target_col = df.columns[-1]

print("Shell 取值：", df[shell_col].unique())

# === 3️⃣ 分组计算每个Shell的极值、均值、标准差 ===
stats_table = df.groupby(shell_col)[target_col].agg(['min', 'max', 'mean', 'std']).reset_index()
print("\n每个Shell的统计结果：\n", stats_table)

plt.figure(figsize=(8,5))
plt.plot(stats_table[shell_col], stats_table['max'], 'o-', label='max', linewidth=2)
plt.plot(stats_table[shell_col], stats_table['min'], 's--', label='min', linewidth=2)
plt.xlabel('Shell')
plt.ylabel('最后一列数值')
plt.title('不同Shell对应的最后一列极值变化')
plt.legend()
plt.grid(True)
plt.show()

# === 5️⃣ ANOVA检验（不同Shell的分布是否显著不同） ===
groups = [g[target_col].values for _, g in df.groupby(shell_col)]
f_stat, p_val = stats.f_oneway(*groups)
print(f"\nANOVA结果: F = {f_stat:.3f}, p = {p_val:.3e}")

if p_val < 0.05:
    print("✅ 结果显著：不同Shell对应的最后一列分布显著不同，说明存在强相关！")
else:
    print("❌ 结果不显著：不同Shell对最后一列影响不显著。")
