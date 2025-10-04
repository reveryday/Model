import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_model_architecture_diagram():
    """创建Physics-Informed Multi-Task Transformer模型架构图"""
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(14, 16))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis('off')
    
    # 定义颜色
    colors = {
        'input': '#E8F4FD',
        'preprocessing': '#D1ECF1', 
        'transformer': '#FFF2CC',
        'physics': '#FFE6CC',
        'output': '#D4EDDA',
        'loss': '#F8D7DA'
    }
    
    # 输入层
    input_box = FancyBboxPatch((1, 18), 8, 1.2, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['input'], 
                               edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(5, 18.6, '输入特征 (8维)', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(5, 18.2, 'Energy, Shell, MFP, MAC_Total, MAC_Incoherent,\nMAC_Coherent, MAC_Photoelectric, MAC_Pair_production', 
            ha='center', va='center', fontsize=9)
    
    # 数据预处理
    preprocess_box = FancyBboxPatch((1, 16.2), 8, 1, 
                                    boxstyle="round,pad=0.1", 
                                    facecolor=colors['preprocessing'], 
                                    edgecolor='black', linewidth=1)
    ax.add_patch(preprocess_box)
    ax.text(5, 16.7, '数据预处理 & 标准化', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # 输入投影层
    projection_box = FancyBboxPatch((1, 14.8), 8, 0.8, 
                                    boxstyle="round,pad=0.1", 
                                    facecolor=colors['transformer'], 
                                    edgecolor='black', linewidth=1)
    ax.add_patch(projection_box)
    ax.text(5, 15.2, '输入投影层 (8 → 256)', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # 位置编码
    pos_encoding_box = FancyBboxPatch((1, 13.4), 8, 0.8, 
                                      boxstyle="round,pad=0.1", 
                                      facecolor=colors['transformer'], 
                                      edgecolor='black', linewidth=1)
    ax.add_patch(pos_encoding_box)
    ax.text(5, 13.8, '位置编码 (Positional Encoding)', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Transformer块 (6层)
    for i in range(6):
        y_pos = 12 - i * 1.5
        
        # Transformer块
        transformer_box = FancyBboxPatch((1.5, y_pos-0.6), 7, 1.2, 
                                         boxstyle="round,pad=0.1", 
                                         facecolor=colors['transformer'], 
                                         edgecolor='black', linewidth=1)
        ax.add_patch(transformer_box)
        ax.text(5, y_pos, f'Transformer Block {i+1}', ha='center', va='center', fontsize=10, fontweight='bold')
        
        # 多头注意力
        attention_box = FancyBboxPatch((2, y_pos-0.4), 2.5, 0.3, 
                                       boxstyle="round,pad=0.05", 
                                       facecolor='white', 
                                       edgecolor='gray', linewidth=0.5)
        ax.add_patch(attention_box)
        ax.text(3.25, y_pos-0.25, '多头注意力 (8头)', ha='center', va='center', fontsize=8)
        
        # 前馈网络
        ff_box = FancyBboxPatch((5.5, y_pos-0.4), 2.5, 0.3, 
                                boxstyle="round,pad=0.05", 
                                facecolor='white', 
                                edgecolor='gray', linewidth=0.5)
        ax.add_patch(ff_box)
        ax.text(6.75, y_pos-0.25, '前馈网络 (1024)', ha='center', va='center', fontsize=8)
    
    # 特征提取器
    feature_extractor_box = FancyBboxPatch((1, 2.8), 8, 0.8, 
                                           boxstyle="round,pad=0.1", 
                                           facecolor=colors['transformer'], 
                                           edgecolor='black', linewidth=1)
    ax.add_patch(feature_extractor_box)
    ax.text(5, 3.2, '特征提取器 (256 → 128)', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # 多任务输出头
    output_heads = [
        'Inf_Flu_BUF', 'Fin_Flu_BUF', 'Inf_Exp_BUF', 
        'Fin_Exp_BUF', 'Inf_Eff_BUF', 'Fin_Eff_BUF'
    ]
    
    for i, head_name in enumerate(output_heads):
        x_pos = 0.5 + i * 1.5
        head_box = FancyBboxPatch((x_pos, 1.5), 1.3, 0.8, 
                                  boxstyle="round,pad=0.05", 
                                  facecolor=colors['output'], 
                                  edgecolor='black', linewidth=1)
        ax.add_patch(head_box)
        ax.text(x_pos + 0.65, 1.9, head_name.replace('_', '\n'), ha='center', va='center', fontsize=8, fontweight='bold')
    
    # 物理约束层
    physics_box = FancyBboxPatch((1, 0.3), 8, 0.8, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor=colors['physics'], 
                                 edgecolor='red', linewidth=2)
    ax.add_patch(physics_box)
    ax.text(5, 0.7, '物理约束层', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(5, 0.4, 'BUF ≥ 1, Finite ≤ Infinite', ha='center', va='center', fontsize=9)
    
    # 添加连接箭头
    arrow_props = dict(arrowstyle='->', lw=2, color='black')
    
    # 主要连接
    connections = [
        ((5, 18), (5, 17.2)),      # 输入 → 预处理
        ((5, 16.2), (5, 15.6)),    # 预处理 → 投影
        ((5, 14.8), (5, 14.2)),    # 投影 → 位置编码
        ((5, 13.4), (5, 12.6)),    # 位置编码 → Transformer1
        ((5, 2.8), (5, 2.3)),      # 特征提取 → 输出头
    ]
    
    for start, end in connections:
        ax.annotate('', xy=end, xytext=start, arrowprops=arrow_props)
    
    # Transformer块之间的连接
    for i in range(5):
        start_y = 12 - i * 1.5 - 0.6
        end_y = 12 - (i + 1) * 1.5 + 0.6
        ax.annotate('', xy=(5, end_y), xytext=(5, start_y), arrowprops=arrow_props)
    
    # 从最后一个Transformer到特征提取器
    ax.annotate('', xy=(5, 3.6), xytext=(5, 4.5), arrowprops=arrow_props)
    
    # 从输出头到物理约束
    for i in range(6):
        x_pos = 0.5 + i * 1.5 + 0.65
        ax.annotate('', xy=(x_pos, 1.1), xytext=(x_pos, 1.5), arrowprops=arrow_props)
    
    # 添加标题
    ax.text(5, 19.5, 'Physics-Informed Multi-Task Transformer', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    ax.text(5, 19.1, '伽马射线屏蔽累积因子预测模型', 
            ha='center', va='center', fontsize=14, fontweight='bold')
    
    # 添加图例
    legend_elements = [
        patches.Patch(color=colors['input'], label='输入层'),
        patches.Patch(color=colors['preprocessing'], label='数据预处理'),
        patches.Patch(color=colors['transformer'], label='Transformer组件'),
        patches.Patch(color=colors['output'], label='输出层'),
        patches.Patch(color=colors['physics'], label='物理约束')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    plt.savefig('model_architecture.png', dpi=300, bbox_inches='tight')
    plt.savefig('model_architecture.svg', format='svg', bbox_inches='tight')
    plt.show()
    
    print("模型架构图已保存为 'model_architecture.png' 和 'model_architecture.svg'")

def create_detailed_transformer_block():
    """创建详细的Transformer块结构图"""
    
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # 输入
    input_box = FancyBboxPatch((4, 10.5), 2, 0.8, 
                               boxstyle="round,pad=0.1", 
                               facecolor='#E8F4FD', 
                               edgecolor='black', linewidth=1)
    ax.add_patch(input_box)
    ax.text(5, 10.9, '输入 X', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # 多头注意力
    attention_box = FancyBboxPatch((1, 8.5), 8, 1.5, 
                                   boxstyle="round,pad=0.1", 
                                   facecolor='#FFF2CC', 
                                   edgecolor='black', linewidth=1)
    ax.add_patch(attention_box)
    ax.text(5, 9.5, '多头自注意力机制', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Q, K, V
    for i, name in enumerate(['Q', 'K', 'V']):
        qkv_box = FancyBboxPatch((1.5 + i * 2.5, 8.7), 1.5, 0.5, 
                                 boxstyle="round,pad=0.05", 
                                 facecolor='white', 
                                 edgecolor='gray', linewidth=0.5)
        ax.add_patch(qkv_box)
        ax.text(2.25 + i * 2.5, 8.95, name, ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 残差连接1
    residual1_box = FancyBboxPatch((4, 7.2), 2, 0.6, 
                                   boxstyle="round,pad=0.1", 
                                   facecolor='#D1ECF1', 
                                   edgecolor='black', linewidth=1)
    ax.add_patch(residual1_box)
    ax.text(5, 7.5, '残差连接 + LayerNorm', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 前馈网络
    ffn_box = FancyBboxPatch((1, 5.5), 8, 1.2, 
                             boxstyle="round,pad=0.1", 
                             facecolor='#FFE6CC', 
                             edgecolor='black', linewidth=1)
    ax.add_patch(ffn_box)
    ax.text(5, 6.3, '前馈神经网络 (FFN)', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(5, 5.9, 'Linear(256→1024) → ReLU → Dropout → Linear(1024→256)', 
            ha='center', va='center', fontsize=9)
    
    # 残差连接2
    residual2_box = FancyBboxPatch((4, 4.2), 2, 0.6, 
                                   boxstyle="round,pad=0.1", 
                                   facecolor='#D1ECF1', 
                                   edgecolor='black', linewidth=1)
    ax.add_patch(residual2_box)
    ax.text(5, 4.5, '残差连接 + LayerNorm', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 输出
    output_box = FancyBboxPatch((4, 2.5), 2, 0.8, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#D4EDDA', 
                                edgecolor='black', linewidth=1)
    ax.add_patch(output_box)
    ax.text(5, 2.9, '输出', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # 添加箭头
    arrow_props = dict(arrowstyle='->', lw=2, color='black')
    
    # 主路径
    ax.annotate('', xy=(5, 10), xytext=(5, 10.5), arrowprops=arrow_props)
    ax.annotate('', xy=(5, 8.5), xytext=(5, 10), arrowprops=arrow_props)
    ax.annotate('', xy=(5, 7.8), xytext=(5, 8.5), arrowprops=arrow_props)
    ax.annotate('', xy=(5, 6.7), xytext=(5, 7.2), arrowprops=arrow_props)
    ax.annotate('', xy=(5, 4.8), xytext=(5, 5.5), arrowprops=arrow_props)
    ax.annotate('', xy=(5, 3.3), xytext=(5, 4.2), arrowprops=arrow_props)
    
    # 残差连接
    # 第一个残差连接
    ax.annotate('', xy=(7.5, 7.5), xytext=(7.5, 9.2), 
                arrowprops=dict(arrowstyle='->', lw=1.5, color='red', 
                               connectionstyle="arc3,rad=0.3"))
    
    # 第二个残差连接
    ax.annotate('', xy=(7.5, 4.5), xytext=(7.5, 6.1), 
                arrowprops=dict(arrowstyle='->', lw=1.5, color='red', 
                               connectionstyle="arc3,rad=0.3"))
    
    # 添加标题
    ax.text(5, 11.5, 'Transformer Block 详细结构', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # 添加注释
    ax.text(8.5, 7.5, '残差连接', ha='center', va='center', fontsize=9, color='red')
    ax.text(8.5, 4.5, '残差连接', ha='center', va='center', fontsize=9, color='red')
    
    plt.tight_layout()
    plt.savefig('transformer_block_detail.png', dpi=300, bbox_inches='tight')
    plt.savefig('transformer_block_detail.svg', format='svg', bbox_inches='tight')
    plt.show()
    
    print("Transformer块详细结构图已保存为 'transformer_block_detail.png' 和 'transformer_block_detail.svg'")

if __name__ == "__main__":
    print("正在生成模型架构图...")
    create_model_architecture_diagram()
    print("\n正在生成Transformer块详细结构图...")
    create_detailed_transformer_block()
    print("\n所有图表生成完成！")