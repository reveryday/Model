#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shell条件混合专家模型架构可视化脚本
绘制详细的模型框架图，显示每一层的输入输出维度
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 模型配置参数（从conf.py获取）
MODEL_CONFIG = {
    'INPUT_DIM': 8,
    'OUTPUT_DIM': 6,
    'NUM_SHELLS': 102,
    'EMBED_DIM': 32,
    'NUM_EXPERTS': 6,
    'EXPERT_HIDDEN_DIM': 256,
    'BACKBONE_D_MODEL': 256,
    'BACKBONE_NHEAD': 8,
    'BACKBONE_LAYERS': 4,
    'BACKBONE_DIM_FEEDFORWARD': 512,
    'BATCH_SIZE': 128  # 示例批次大小
}

def create_model_architecture_diagram():
    """创建模型架构图"""
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(20, 16))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 24)
    ax.axis('off')
    
    # 定义颜色方案
    colors = {
        'input': '#E3F2FD',      # 浅蓝色 - 输入
        'embedding': '#F3E5F5',   # 浅紫色 - 嵌入层
        'backbone': '#E8F5E8',    # 浅绿色 - 主干网络
        'moe': '#FFF3E0',        # 浅橙色 - MoE层
        'expert': '#FFEBEE',     # 浅红色 - 专家网络
        'gate': '#F1F8E9',       # 浅黄绿色 - 门控网络
        'output': '#E0F2F1',     # 浅青色 - 输出层
        'connection': '#757575'   # 灰色 - 连接线
    }
    
    # 批次大小
    batch_size = MODEL_CONFIG['BATCH_SIZE']
    
    # 1. 输入层
    input_box = draw_layer_box(ax, 2, 22, 3, 1.5, 
                              f"输入层\n[{batch_size}, {MODEL_CONFIG['INPUT_DIM']}]", 
                              colors['input'])
    
    # 2. Shell提取和嵌入
    shell_extract_box = draw_layer_box(ax, 0.5, 19.5, 2.5, 1, 
                                      f"Shell索引提取\n[{batch_size}]", 
                                      colors['input'])
    
    shell_embed_box = draw_layer_box(ax, 0.5, 17.5, 2.5, 1.5, 
                                    f"Shell嵌入层\n[{batch_size}, {MODEL_CONFIG['EMBED_DIM']}]", 
                                    colors['embedding'])
    
    # 3. 共享主干网络 (Transformer)
    # 输入投影
    input_proj_box = draw_layer_box(ax, 6, 20, 3, 1.5, 
                                   f"输入投影\n[{batch_size}, {MODEL_CONFIG['BACKBONE_D_MODEL']}]", 
                                   colors['backbone'])
    
    # 位置编码
    pos_enc_box = draw_layer_box(ax, 6, 18, 3, 1, 
                                f"位置编码\n[{batch_size}, 1, {MODEL_CONFIG['BACKBONE_D_MODEL']}]", 
                                colors['backbone'])
    
    # Transformer编码器
    transformer_box = draw_layer_box(ax, 6, 15.5, 3, 2, 
                                    f"Transformer编码器\n{MODEL_CONFIG['BACKBONE_LAYERS']}层\n"
                                    f"[{batch_size}, 1, {MODEL_CONFIG['BACKBONE_D_MODEL']}]", 
                                    colors['backbone'])
    
    # 输出投影
    output_proj_box = draw_layer_box(ax, 6, 13, 3, 1.5, 
                                    f"输出投影\n[{batch_size}, {MODEL_CONFIG['BACKBONE_D_MODEL']//2}]", 
                                    colors['backbone'])
    
    # 4. 混合专家层 (MoE)
    # 门控网络
    gate_box = draw_layer_box(ax, 11, 15, 3.5, 2, 
                             f"门控网络\n输入: [{batch_size}, {MODEL_CONFIG['BACKBONE_D_MODEL']//2 + MODEL_CONFIG['EMBED_DIM']}]\n"
                             f"输出: [{batch_size}, {MODEL_CONFIG['NUM_EXPERTS']}]", 
                             colors['gate'])
    
    # 专家网络
    expert_y_positions = np.linspace(18, 8, MODEL_CONFIG['NUM_EXPERTS'])
    expert_boxes = []
    for i, y_pos in enumerate(expert_y_positions):
        expert_box = draw_layer_box(ax, 15.5, y_pos, 3, 1.5, 
                                   f"专家 {i+1}\n[{batch_size}, {MODEL_CONFIG['EXPERT_HIDDEN_DIM']}]", 
                                   colors['expert'])
        expert_boxes.append(expert_box)
    
    # MoE输出
    moe_output_box = draw_layer_box(ax, 11, 10, 3.5, 1.5, 
                                   f"MoE输出\n[{batch_size}, {MODEL_CONFIG['EXPERT_HIDDEN_DIM']}]", 
                                   colors['moe'])
    
    # 5. 多任务输出头
    task_heads_box = draw_layer_box(ax, 6, 7, 3, 2, 
                                   f"多任务输出头\n{MODEL_CONFIG['OUTPUT_DIM']}个任务\n"
                                   f"[{batch_size}, {MODEL_CONFIG['OUTPUT_DIM']}]", 
                                   colors['output'])
    
    # 6. 最终输出
    final_output_box = draw_layer_box(ax, 6, 4, 3, 1.5, 
                                     f"最终输出\n[{batch_size}, {MODEL_CONFIG['OUTPUT_DIM']}]", 
                                     colors['output'])
    
    # 绘制连接线
    draw_connections(ax, colors['connection'])
    
    # 添加标题和说明
    ax.text(10, 23.5, 'Shell条件混合专家模型架构图', 
            fontsize=20, fontweight='bold', ha='center')
    
    # 添加图例
    add_legend(ax, colors)
    
    # 添加详细说明
    add_detailed_info(ax)
    
    plt.tight_layout()
    return fig

def draw_layer_box(ax, x, y, width, height, text, color):
    """绘制层的方框"""
    box = FancyBboxPatch((x, y), width, height,
                        boxstyle="round,pad=0.1",
                        facecolor=color,
                        edgecolor='black',
                        linewidth=1.5)
    ax.add_patch(box)
    
    # 添加文本
    ax.text(x + width/2, y + height/2, text,
           ha='center', va='center', fontsize=10, fontweight='bold')
    
    return box

def draw_connections(ax, color):
    """绘制层之间的连接线"""
    # 定义连接点
    connections = [
        # 从输入到Shell提取
        ((3.5, 22), (1.75, 20.5)),
        # 从Shell提取到Shell嵌入
        ((1.75, 19.5), (1.75, 19)),
        # 从输入到输入投影
        ((3.5, 22.75), (7.5, 21.5)),
        # 主干网络内部连接
        ((7.5, 20), (7.5, 19)),
        ((7.5, 18), (7.5, 17.5)),
        ((7.5, 15.5), (7.5, 14.5)),
        # 从输出投影到门控网络
        ((9, 13.75), (12.75, 16)),
        # 从Shell嵌入到门控网络
        ((3, 18.25), (12.75, 16)),
        # 从输出投影到专家网络
        ((9, 13.75), (15.5, 13)),
        # 从门控网络到MoE输出
        ((12.75, 15), (12.75, 11.5)),
        # 从专家网络到MoE输出
        ((17, 13), (14.5, 10.75)),
        # 从MoE输出到多任务头
        ((12.75, 10), (7.5, 9)),
        # 从多任务头到最终输出
        ((7.5, 7), (7.5, 5.5))
    ]
    
    for start, end in connections:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', color=color, lw=2))

def add_legend(ax, colors):
    """添加图例"""
    legend_elements = [
        ('输入层', colors['input']),
        ('嵌入层', colors['embedding']),
        ('主干网络', colors['backbone']),
        ('门控网络', colors['gate']),
        ('专家网络', colors['expert']),
        ('MoE层', colors['moe']),
        ('输出层', colors['output'])
    ]
    
    legend_x = 0.5
    legend_y = 14
    
    ax.text(legend_x, legend_y + 1, '图例:', fontsize=12, fontweight='bold')
    
    for i, (label, color) in enumerate(legend_elements):
        y_pos = legend_y - i * 0.5
        # 绘制小方块
        rect = patches.Rectangle((legend_x, y_pos - 0.15), 0.3, 0.3, 
                               facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        # 添加标签
        ax.text(legend_x + 0.4, y_pos, label, fontsize=10, va='center')

def add_detailed_info(ax):
    """添加详细信息"""
    info_text = f"""
模型详细参数:
• 输入维度: {MODEL_CONFIG['INPUT_DIM']} (Energy, Shell, MFP, MAC_Total, MAC_Incoherent, MAC_Coherent, MAC_Photoelectric, MAC_Pair_production)
• 输出维度: {MODEL_CONFIG['OUTPUT_DIM']} (Inf_Flu_BUF, Fin_Flu_BUF, Inf_Exp_BUF, Fin_Exp_BUF, Inf_Eff_BUF, Fin_Eff_BUF)
• Shell数量: {MODEL_CONFIG['NUM_SHELLS']} (0-101)
• Shell嵌入维度: {MODEL_CONFIG['EMBED_DIM']}
• 专家网络数量: {MODEL_CONFIG['NUM_EXPERTS']}
• 专家隐藏层维度: {MODEL_CONFIG['EXPERT_HIDDEN_DIM']}
• Transformer维度: {MODEL_CONFIG['BACKBONE_D_MODEL']}
• Transformer头数: {MODEL_CONFIG['BACKBONE_NHEAD']}
• Transformer层数: {MODEL_CONFIG['BACKBONE_LAYERS']}
• 前馈网络维度: {MODEL_CONFIG['BACKBONE_DIM_FEEDFORWARD']}
    """
    
    ax.text(0.5, 6, info_text.strip(), fontsize=9, va='top', 
           bbox=dict(boxstyle="round,pad=0.5", facecolor='#F5F5F5', alpha=0.8))

def create_data_flow_diagram():
    """创建数据流图"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # 数据流步骤
    steps = [
        (2, 10.5, "输入数据\n[B, 8]", '#E3F2FD'),
        (2, 8.5, "Shell索引提取\n[B]", '#F3E5F5'),
        (2, 6.5, "Shell嵌入\n[B, 32]", '#F3E5F5'),
        (8, 10.5, "输入投影\n[B, 256]", '#E8F5E8'),
        (8, 8.5, "Transformer\n[B, 128]", '#E8F5E8'),
        (14, 9.5, "门控权重\n[B, 6]", '#F1F8E9'),
        (14, 7.5, "专家输出\n[B, 6, 256]", '#FFEBEE'),
        (8, 6.5, "MoE输出\n[B, 256]", '#FFF3E0'),
        (8, 4.5, "多任务头\n[B, 6]", '#E0F2F1'),
        (8, 2.5, "最终输出\n[B, 6]", '#E0F2F1')
    ]
    
    # 绘制步骤
    for x, y, text, color in steps:
        draw_layer_box(ax, x-1, y-0.5, 2, 1, text, color)
    
    # 绘制数据流箭头
    flow_connections = [
        ((3, 10.5), (7, 10.5)),  # 输入到投影
        ((3, 10), (3, 9)),       # 输入到Shell提取
        ((3, 8), (3, 7)),        # Shell提取到嵌入
        ((8, 10), (8, 9)),       # 投影到Transformer
        ((9, 8.5), (13, 9.5)),   # Transformer到门控
        ((9, 8.5), (13, 7.5)),   # Transformer到专家
        ((4, 6.5), (13, 9.5)),   # Shell嵌入到门控
        ((14, 7), (9, 6.5)),     # 专家到MoE
        ((13, 9.5), (9, 6.5)),   # 门控到MoE
        ((8, 6), (8, 5)),        # MoE到多任务头
        ((8, 4), (8, 3))         # 多任务头到输出
    ]
    
    for start, end in flow_connections:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', color='#757575', lw=2))
    
    ax.text(8, 11.5, '数据流图 (B = Batch Size)', 
            fontsize=16, fontweight='bold', ha='center')
    
    return fig

def create_expert_detail_diagram():
    """创建专家网络详细结构图"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 专家网络结构
    expert_layers = [
        (2, 8, "输入\n[B, 128]", '#E3F2FD'),
        (2, 6.5, "Linear(128→256)\n+ LayerNorm + ReLU", '#FFEBEE'),
        (2, 5, "Dropout(0.1)", '#FFEBEE'),
        (2, 3.5, "Linear(256→128)\n+ LayerNorm + ReLU", '#FFEBEE'),
        (2, 2, "Dropout(0.1)", '#FFEBEE'),
        (2, 0.5, "Linear(128→256)\n输出", '#E0F2F1')
    ]
    
    # 门控网络结构
    gate_layers = [
        (8, 8, "输入\n[B, 128+32]", '#E3F2FD'),
        (8, 6.5, "Linear(160→128)\n+ LayerNorm + ReLU", '#F1F8E9'),
        (8, 5, "Dropout(0.1)", '#F1F8E9'),
        (8, 3.5, "Linear(128→64)\n+ LayerNorm + ReLU", '#F1F8E9'),
        (8, 2, "Dropout(0.1)", '#F1F8E9'),
        (8, 0.5, "Linear(64→6)\n+ Softmax", '#E0F2F1')
    ]
    
    # 绘制专家网络
    ax.text(2, 9, '专家网络结构', fontsize=14, fontweight='bold', ha='center')
    for x, y, text, color in expert_layers:
        draw_layer_box(ax, x-1, y-0.3, 2, 0.6, text, color)
    
    # 绘制门控网络
    ax.text(8, 9, '门控网络结构', fontsize=14, fontweight='bold', ha='center')
    for x, y, text, color in gate_layers:
        draw_layer_box(ax, x-1, y-0.3, 2, 0.6, text, color)
    
    # 绘制连接
    for i in range(len(expert_layers)-1):
        y1 = expert_layers[i][1] - 0.3
        y2 = expert_layers[i+1][1] + 0.3
        ax.annotate('', xy=(2, y2), xytext=(2, y1),
                   arrowprops=dict(arrowstyle='->', color='#757575', lw=2))
    
    for i in range(len(gate_layers)-1):
        y1 = gate_layers[i][1] - 0.3
        y2 = gate_layers[i+1][1] + 0.3
        ax.annotate('', xy=(8, y2), xytext=(8, y1),
                   arrowprops=dict(arrowstyle='->', color='#757575', lw=2))
    
    return fig

def main():
    """主函数"""
    print("正在生成Shell条件混合专家模型架构图...")
    
    # 创建输出目录
    output_dir = Path(".")
    
    # 1. 生成主要架构图
    print("1. 生成主要架构图...")
    fig1 = create_model_architecture_diagram()
    arch_path = output_dir / "shell_moe_architecture.png"
    fig1.savefig(arch_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   架构图已保存: {arch_path}")
    
    # 2. 生成数据流图
    print("2. 生成数据流图...")
    fig2 = create_data_flow_diagram()
    flow_path = output_dir / "shell_moe_dataflow.png"
    fig2.savefig(flow_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   数据流图已保存: {flow_path}")
    
    # 3. 生成专家网络详细图
    print("3. 生成专家网络详细图...")
    fig3 = create_expert_detail_diagram()
    expert_path = output_dir / "shell_moe_expert_details.png"
    fig3.savefig(expert_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   专家网络详细图已保存: {expert_path}")
    
    print("\n✅ 所有架构图生成完成!")
    print(f"📊 主要架构图: {arch_path}")
    print(f"🔄 数据流图: {flow_path}")
    print(f"🔧 专家网络详细图: {expert_path}")
    
    # 显示图表
    plt.show()

if __name__ == "__main__":
    main()