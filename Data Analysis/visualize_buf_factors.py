import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_and_analyze_data(csv_path):
    """加载并分析数据"""
    print(f"正在加载数据: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 后六列累计因子
    buf_columns = ['Inf_Flu_BUF', 'Fin_Flu_BUF', 'Inf_Exp_BUF', 
                   'Fin_Exp_BUF', 'Inf_Eff_BUF', 'Fin_Eff_BUF']
    
    print(f"数据形状: {df.shape}")
    print(f"累计因子列: {buf_columns}")
    
    # 基本统计信息
    buf_data = df[buf_columns]
    print("\n累计因子基本统计信息:")
    print(buf_data.describe())
    
    return df, buf_data, buf_columns

def create_distribution_plots(buf_data, buf_columns, output_dir):
    """创建分布图"""
    
    # 1. 箱线图 - 显示数值范围和异常值
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 2, 1)
    box_plot = plt.boxplot([buf_data[col] for col in buf_columns], 
                          labels=[col.replace('_', '\n') for col in buf_columns],
                          patch_artist=True)
    
    # 设置箱线图颜色
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'lightgray']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.title('累计因子箱线图分布', fontsize=14, fontweight='bold')
    plt.ylabel('数值', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 2. 小提琴图 - 显示数据密度分布
    plt.subplot(2, 2, 2)
    violin_data = [buf_data[col] for col in buf_columns]
    parts = plt.violinplot(violin_data, positions=range(1, len(buf_columns)+1))
    
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    
    plt.title('累计因子小提琴图分布', fontsize=14, fontweight='bold')
    plt.ylabel('数值', fontsize=12)
    plt.xticks(range(1, len(buf_columns)+1), [col.replace('_', '\n') for col in buf_columns], rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 3. 直方图矩阵
    plt.subplot(2, 2, 3)
    # 计算每个因子的数值范围
    ranges = []
    for col in buf_columns:
        min_val = buf_data[col].min()
        max_val = buf_data[col].max()
        ranges.append(f'{col}:\n[{min_val:.2f}, {max_val:.2f}]')
    
    # 创建范围条形图
    y_pos = np.arange(len(buf_columns))
    min_vals = [buf_data[col].min() for col in buf_columns]
    max_vals = [buf_data[col].max() for col in buf_columns]
    
    plt.barh(y_pos, max_vals, color=colors, alpha=0.7, label='最大值')
    plt.barh(y_pos, min_vals, color='darkblue', alpha=0.5, label='最小值')
    
    plt.yticks(y_pos, [col.replace('_', '\n') for col in buf_columns])
    plt.xlabel('数值范围', fontsize=12)
    plt.title('累计因子数值范围', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. 统计摘要表格
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    # 创建统计摘要
    stats_data = []
    for col in buf_columns:
        stats_data.append([
            col.replace('_', ' '),
            f"{buf_data[col].min():.3f}",
            f"{buf_data[col].max():.3f}",
            f"{buf_data[col].mean():.3f}",
            f"{buf_data[col].std():.3f}"
        ])
    
    table = plt.table(cellText=stats_data,
                     colLabels=['累计因子', '最小值', '最大值', '平均值', '标准差'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # 设置表格样式
    for i in range(len(buf_columns) + 1):
        for j in range(5):
            if i == 0:  # 表头
                table[(i, j)].set_facecolor('#4CAF50')
                table[(i, j)].set_text_props(weight='bold', color='white')
            else:
                table[(i, j)].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    
    plt.title('累计因子统计摘要', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # 保存图片
    output_path = output_dir / 'buf_factors_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"分布图已保存: {output_path}")
    
    return output_path

def create_detailed_histograms(buf_data, buf_columns, output_dir):
    """创建详细的直方图"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    colors = ['skyblue', 'lightgreen', 'salmon', 'gold', 'plum', 'lightcyan']
    
    for i, col in enumerate(buf_columns):
        ax = axes[i]
        
        # 绘制直方图
        n, bins, patches = ax.hist(buf_data[col], bins=50, color=colors[i], 
                                  alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # 添加统计信息
        mean_val = buf_data[col].mean()
        std_val = buf_data[col].std()
        min_val = buf_data[col].min()
        max_val = buf_data[col].max()
        
        # 添加垂直线显示平均值
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'平均值: {mean_val:.3f}')
        
        # 设置标题和标签
        ax.set_title(f'{col.replace("_", " ")}\n范围: [{min_val:.3f}, {max_val:.3f}]', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('数值', fontsize=10)
        ax.set_ylabel('频次', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 添加统计文本
        stats_text = f'均值: {mean_val:.3f}\n标准差: {std_val:.3f}\n样本数: {len(buf_data[col])}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               fontsize=8)
    
    plt.suptitle('累计因子详细分布直方图', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存图片
    output_path = output_dir / 'buf_factors_histograms.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"详细直方图已保存: {output_path}")
    
    return output_path

def create_correlation_heatmap(buf_data, buf_columns, output_dir):
    """创建相关性热力图"""
    
    plt.figure(figsize=(10, 8))
    
    # 计算相关性矩阵
    correlation_matrix = buf_data[buf_columns].corr()
    
    # 创建热力图
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, 
                mask=mask,
                annot=True, 
                cmap='coolwarm', 
                center=0,
                square=True,
                fmt='.3f',
                cbar_kws={'label': '相关系数'})
    
    plt.title('累计因子相关性热力图', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # 保存图片
    output_path = output_dir / 'buf_factors_correlation.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"相关性热力图已保存: {output_path}")
    
    return output_path

def main():
    """主函数"""
    # 设置路径
    data_path = Path("data.csv")
    output_dir = Path(".")
    
    if not data_path.exists():
        print(f"错误: 找不到数据文件 {data_path}")
        return
    
    try:
        # 加载和分析数据
        df, buf_data, buf_columns = load_and_analyze_data(data_path)
        
        print("\n开始生成可视化图表...")
        
        # 创建分布图
        dist_path = create_distribution_plots(buf_data, buf_columns, output_dir)
        
        # 创建详细直方图
        hist_path = create_detailed_histograms(buf_data, buf_columns, output_dir)
        
        # 创建相关性热力图
        corr_path = create_correlation_heatmap(buf_data, buf_columns, output_dir)
        
        print(f"\n✅ 所有图表生成完成!")
        print(f"📊 分布图: {dist_path}")
        print(f"📈 详细直方图: {hist_path}")
        print(f"🔥 相关性热力图: {corr_path}")
        
        # 显示图表
        plt.show()
        
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()