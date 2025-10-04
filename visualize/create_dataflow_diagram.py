import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Arrow
import numpy as np

def create_physics_constraint_diagram():
    """创建物理约束和损失函数的详细图"""
    
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 左图：物理约束
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('物理约束机制', fontsize=16, fontweight='bold', pad=20)
    
    # 模型输出
    output_box = FancyBboxPatch((1, 8), 8, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#D4EDDA', 
                                edgecolor='black', linewidth=2)
    ax1.add_patch(output_box)
    ax1.text(5, 8.5, '模型原始输出 (6个累积因子)', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # 物理约束检查
    constraint_boxes = [
        {'pos': (0.5, 6), 'text': 'BUF ≥ 1\n约束检查', 'color': '#FFE6CC'},
        {'pos': (5, 6), 'text': 'Finite ≤ Infinite\n约束检查', 'color': '#FFE6CC'},
    ]
    
    for box in constraint_boxes:
        constraint_box = FancyBboxPatch(box['pos'], 3, 1.2, 
                                        boxstyle="round,pad=0.1", 
                                        facecolor=box['color'], 
                                        edgecolor='red', linewidth=2)
        ax1.add_patch(constraint_box)
        ax1.text(box['pos'][0] + 1.5, box['pos'][1] + 0.6, box['text'], 
                ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 约束应用
    apply_box = FancyBboxPatch((2, 4), 6, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor='#F8D7DA', 
                               edgecolor='red', linewidth=2)
    ax1.add_patch(apply_box)
    ax1.text(5, 4.5, '物理约束应用层', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # 最终输出
    final_box = FancyBboxPatch((1, 2), 8, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor='#D1ECF1', 
                               edgecolor='black', linewidth=2)
    ax1.add_patch(final_box)
    ax1.text(5, 2.5, '物理一致的累积因子输出', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # 添加箭头
    arrow_props = dict(arrowstyle='->', lw=2, color='black')
    ax1.annotate('', xy=(2, 7.2), xytext=(3, 8), arrowprops=arrow_props)
    ax1.annotate('', xy=(6.5, 7.2), xytext=(7, 8), arrowprops=arrow_props)
    ax1.annotate('', xy=(5, 5), xytext=(2, 6), arrowprops=arrow_props)
    ax1.annotate('', xy=(5, 5), xytext=(6.5, 6), arrowprops=arrow_props)
    ax1.annotate('', xy=(5, 3), xytext=(5, 4), arrowprops=arrow_props)
    
    # 右图：损失函数
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('Physics-Informed Loss Function', fontsize=16, fontweight='bold', pad=20)
    
    # 预测值和真实值
    pred_box = FancyBboxPatch((0.5, 8.5), 3, 0.8, 
                              boxstyle="round,pad=0.1", 
                              facecolor='#E8F4FD', 
                              edgecolor='black', linewidth=1)
    ax2.add_patch(pred_box)
    ax2.text(2, 8.9, '预测值', ha='center', va='center', fontsize=11, fontweight='bold')
    
    true_box = FancyBboxPatch((6, 8.5), 3, 0.8, 
                              boxstyle="round,pad=0.1", 
                              facecolor='#E8F4FD', 
                              edgecolor='black', linewidth=1)
    ax2.add_patch(true_box)
    ax2.text(7.5, 8.9, '真实值', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # MSE损失
    mse_box = FancyBboxPatch((3, 7), 4, 0.8, 
                             boxstyle="round,pad=0.1", 
                             facecolor='#FFF2CC', 
                             edgecolor='black', linewidth=1)
    ax2.add_patch(mse_box)
    ax2.text(5, 7.4, 'MSE Loss', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # 物理约束损失
    physics_loss_boxes = [
        {'pos': (0.5, 5.5), 'text': 'BUF ≥ 1\n约束损失', 'color': '#FFE6CC'},
        {'pos': (5.5, 5.5), 'text': 'Finite ≤ Infinite\n约束损失', 'color': '#FFE6CC'},
    ]
    
    for box in physics_loss_boxes:
        loss_box = FancyBboxPatch(box['pos'], 3.5, 1, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor=box['color'], 
                                  edgecolor='red', linewidth=1)
        ax2.add_patch(loss_box)
        ax2.text(box['pos'][0] + 1.75, box['pos'][1] + 0.5, box['text'], 
                ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 总损失
    total_loss_box = FancyBboxPatch((2, 3.5), 6, 1, 
                                    boxstyle="round,pad=0.1", 
                                    facecolor='#F8D7DA', 
                                    edgecolor='red', linewidth=2)
    ax2.add_patch(total_loss_box)
    ax2.text(5, 4, 'Total Loss = MSE + λ₁×Physics_Loss₁ + λ₂×Physics_Loss₂', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 添加箭头
    ax2.annotate('', xy=(4, 7.8), xytext=(2, 8.5), arrowprops=arrow_props)
    ax2.annotate('', xy=(6, 7.8), xytext=(7.5, 8.5), arrowprops=arrow_props)
    ax2.annotate('', xy=(3, 4.5), xytext=(5, 7), arrowprops=arrow_props)
    ax2.annotate('', xy=(4, 4.5), xytext=(2.25, 5.5), arrowprops=arrow_props)
    ax2.annotate('', xy=(6, 4.5), xytext=(7.25, 5.5), arrowprops=arrow_props)
    
    plt.tight_layout()
    plt.savefig('physics_constraint_diagram.png', dpi=300, bbox_inches='tight')
    plt.savefig('physics_constraint_diagram.svg', format='svg', bbox_inches='tight')
    plt.show()
    
    print("物理约束和损失函数图已保存")

def create_data_flow_diagram():
    """创建完整的数据流图"""
    
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # 数据输入
    data_box = FancyBboxPatch((1, 10), 3, 1.5, 
                              boxstyle="round,pad=0.1", 
                              facecolor='#E8F4FD', 
                              edgecolor='black', linewidth=2)
    ax.add_patch(data_box)
    ax.text(2.5, 10.75, '原始数据\n(90,168 × 14)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 数据分割
    split_boxes = [
        {'pos': (6, 10.5), 'text': '输入特征\n(8维)', 'color': '#D1ECF1'},
        {'pos': (10, 10.5), 'text': '输出标签\n(6维)', 'color': '#D4EDDA'},
    ]
    
    for box in split_boxes:
        split_box = FancyBboxPatch(box['pos'], 2.5, 1, 
                                   boxstyle="round,pad=0.1", 
                                   facecolor=box['color'], 
                                   edgecolor='black', linewidth=1)
        ax.add_patch(split_box)
        ax.text(box['pos'][0] + 1.25, box['pos'][1] + 0.5, box['text'], 
                ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 数据预处理
    preprocess_box = FancyBboxPatch((6, 8.5), 2.5, 1, 
                                    boxstyle="round,pad=0.1", 
                                    facecolor='#FFF2CC', 
                                    edgecolor='black', linewidth=1)
    ax.add_patch(preprocess_box)
    ax.text(7.25, 9, '标准化\n(StandardScaler)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 模型
    model_box = FancyBboxPatch((5, 6), 4, 2, 
                               boxstyle="round,pad=0.1", 
                               facecolor='#FFE6CC', 
                               edgecolor='black', linewidth=2)
    ax.add_patch(model_box)
    ax.text(7, 7, 'Physics-Informed\nTransformer', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # 预测输出
    pred_box = FancyBboxPatch((5, 4), 4, 1, 
                              boxstyle="round,pad=0.1", 
                              facecolor='#D4EDDA', 
                              edgecolor='black', linewidth=1)
    ax.add_patch(pred_box)
    ax.text(7, 4.5, '预测的累积因子', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # 损失计算
    loss_box = FancyBboxPatch((5, 2), 4, 1, 
                              boxstyle="round,pad=0.1", 
                              facecolor='#F8D7DA', 
                              edgecolor='red', linewidth=2)
    ax.add_patch(loss_box)
    ax.text(7, 2.5, 'Physics-Informed Loss', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # 反向传播
    backprop_box = FancyBboxPatch((1, 2), 3, 1, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor='#E6E6FA', 
                                  edgecolor='purple', linewidth=2)
    ax.add_patch(backprop_box)
    ax.text(2.5, 2.5, '反向传播\n参数更新', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 添加箭头
    arrow_props = dict(arrowstyle='->', lw=2, color='black')
    
    # 数据流
    ax.annotate('', xy=(6, 11), xytext=(4, 10.75), arrowprops=arrow_props)
    ax.annotate('', xy=(10, 11), xytext=(4, 10.75), arrowprops=arrow_props)
    ax.annotate('', xy=(7.25, 8.5), xytext=(7.25, 10.5), arrowprops=arrow_props)
    ax.annotate('', xy=(7, 8), xytext=(7.25, 8.5), arrowprops=arrow_props)
    ax.annotate('', xy=(7, 5), xytext=(7, 6), arrowprops=arrow_props)
    ax.annotate('', xy=(7, 3), xytext=(7, 4), arrowprops=arrow_props)
    
    # 标签到损失
    ax.annotate('', xy=(9, 2.5), xytext=(11.25, 10.5), 
                arrowprops=dict(arrowstyle='->', lw=1.5, color='green', 
                               connectionstyle="arc3,rad=0.3"))
    
    # 反向传播
    ax.annotate('', xy=(4, 2.5), xytext=(5, 2.5), arrowprops=arrow_props)
    ax.annotate('', xy=(2.5, 3), xytext=(2.5, 6), 
                arrowprops=dict(arrowstyle='->', lw=1.5, color='purple', 
                               connectionstyle="arc3,rad=-0.3"))
    
    # 添加标题
    ax.text(7, 11.5, '伽马射线屏蔽累积因子预测模型数据流图', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # 添加注释
    ax.text(11.5, 8, '真实标签', ha='center', va='center', fontsize=9, color='green')
    ax.text(0.5, 4, '梯度反向传播', ha='center', va='center', fontsize=9, color='purple', rotation=90)
    
    plt.tight_layout()
    plt.savefig('data_flow_diagram.png', dpi=300, bbox_inches='tight')
    plt.savefig('data_flow_diagram.svg', format='svg', bbox_inches='tight')
    plt.show()
    
    print("数据流图已保存")

if __name__ == "__main__":
    print("正在生成物理约束和损失函数图...")
    create_physics_constraint_diagram()
    print("\n正在生成数据流图...")
    create_data_flow_diagram()
    print("\n所有图表生成完成！")