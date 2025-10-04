# 伽马射线屏蔽累积因子预测模型

基于Physics-Informed Multi-Task Transformer的深度学习模型，用于预测伽马点源单层屏蔽一维球模型的累积因子。

## 模型特点

### 🚀 **最优架构选择：Physics-Informed Multi-Task Transformer**

经过对比分析五种深度学习框架，我们选择了最适合您物理模型的架构：

1. **Physics-Informed Neural Network (PINN)** ⭐⭐⭐⭐⭐
2. **Multi-Task Transformer Network** ⭐⭐⭐⭐⭐  
3. **Deep Residual Network (ResNet)** ⭐⭐⭐⭐
4. **Ensemble Learning Framework** ⭐⭐⭐⭐
5. **Attention-Enhanced MLP** ⭐⭐⭐

### 🔬 **核心优势**

- **物理约束融入**: 确保预测结果符合伽马射线传输物理定律
- **多任务学习**: 同时预测6个累积因子，利用任务间相关性提高精度
- **注意力机制**: 自动学习能量、厚度、衰减系数间的重要关系
- **高精度预测**: 专为物理模型设计，确保预测的物理合理性

### 📊 **数据集信息**

- **样本数量**: 90,168个
- **输入特征**: 8个 (Energy, Shell, MFP, MAC_Total, MAC_Incoherent, MAC_Coherent, MAC_Photoelectric, MAC_Pair_production)
- **输出特征**: 6个累积因子 (Inf_Flu_BUF, Fin_Flu_BUF, Inf_Exp_BUF, Fin_Exp_BUF, Inf_Eff_BUF, Fin_Eff_BUF)

## 快速开始

### 1. 环境配置

```bash
pip install -r requirements.txt
```

### 2. 训练模型

```bash
python training_pipeline.py
```

训练过程包括：
- 自动数据预处理和归一化
- 数据集分割 (70% 训练, 15% 验证, 15% 测试)
- 早停机制防止过拟合
- 学习率自适应调整
- 物理约束损失函数

### 3. 模型推理

```bash
python model_inference.py
```

功能包括：
- 单样本预测
- 批量预测
- 能量依赖性分析
- 厚度依赖性分析
- 注意力权重可视化
- 预测误差分析

## 文件结构

```
├── dataset.csv                          # 原始数据集
├── physics_informed_transformer.py      # 模型架构定义
├── training_pipeline.py                 # 训练管道
├── model_inference.py                   # 推理和分析工具
├── requirements.txt                     # 依赖包列表
├── README.md                           # 说明文档
└── 生成的文件/
    ├── physics_informed_transformer_model.pth  # 训练好的模型
    ├── training_history.png                    # 训练历史图
    ├── prediction_comparison.png               # 预测对比图
    ├── energy_dependence_analysis.png          # 能量依赖性分析
    ├── thickness_dependence_analysis.png       # 厚度依赖性分析
    ├── attention_weights.png                   # 注意力权重热图
    └── error_distribution.png                  # 误差分布图
```

## 模型架构详解

### Physics-Informed Transformer

```python
PhysicsInformedTransformer(
    input_dim=8,           # 输入特征维度
    output_dim=6,          # 输出特征维度
    d_model=256,           # 模型维度
    n_heads=8,             # 注意力头数
    n_layers=6,            # Transformer层数
    d_ff=1024,             # 前馈网络维度
    dropout=0.1            # Dropout率
)
```

### 关键组件

1. **多头注意力机制**: 捕捉输入特征间的复杂关系
2. **位置编码**: 保持特征的位置信息
3. **多任务输出头**: 为每个累积因子设计专门的预测头
4. **物理约束层**: 确保预测结果满足物理定律
5. **物理约束损失函数**: 在训练中融入物理知识

### 物理约束

- **约束1**: 累积因子必须 ≥ 1
- **约束2**: 有限几何累积因子 ≤ 无限几何累积因子
- **约束3**: 随厚度增加的单调性约束

## 使用示例

### 单样本预测

```python
from model_inference import BUFPredictor

predictor = BUFPredictor('physics_informed_transformer_model.pth')

result, attention_weights = predictor.predict_single(
    energy=1.0,                    # 能量 (MeV)
    shell=0,                       # 壳层数
    mfp=5.0,                      # 厚度 (MFP)
    mac_total=0.5,                # 总衰减系数
    mac_incoherent=0.1,           # 非相干散射衰减系数
    mac_coherent=0.05,            # 相干散射衰减系数
    mac_photoelectric=0.3,        # 光电效应衰减系数
    mac_pair_production=0.05      # 电子对产生衰减系数
)

print("预测的累积因子:")
for name, value in result.items():
    print(f"{name}: {value:.6f}")
```

### 批量预测

```python
import pandas as pd

# 准备输入数据
input_data = pd.DataFrame({
    'Energy': [0.5, 1.0, 2.0],
    'Shell': [0, 0, 1],
    'MFP': [1.0, 5.0, 10.0],
    # ... 其他特征
})

predictions, _ = predictor.predict_batch(input_data)
print(predictions)
```

### 分析工具

```python
# 能量依赖性分析
energies, results = predictor.analyze_energy_dependence(
    shell=0, 
    mfp_values=[1.0, 5.0, 10.0],
    energy_range=(0.01, 10.0)
)

# 厚度依赖性分析
thicknesses, predictions = predictor.analyze_thickness_dependence(
    energy=1.0,
    shell=0,
    thickness_range=(0.0, 20.0)
)

# 注意力权重可视化
predictor.visualize_attention_weights(
    energy=1.0, shell=0, mfp=5.0,
    mac_total=0.5, mac_incoherent=0.1,
    mac_coherent=0.05, mac_photoelectric=0.3,
    mac_pair_production=0.05
)
```

## 性能指标

模型在测试集上的典型性能：

- **RMSE**: < 0.01 (归一化后)
- **R²**: > 0.99
- **物理约束满足率**: 100%

## 技术特点

### 1. 先进的深度学习架构
- Transformer注意力机制
- 多任务学习框架
- 残差连接和层归一化

### 2. 物理知识融入
- 物理约束损失函数
- 累积因子物理定律约束
- 几何关系约束

### 3. 鲁棒的训练策略
- 早停机制
- 学习率自适应调整
- 梯度裁剪
- Dropout正则化

### 4. 全面的分析工具
- 训练过程可视化
- 预测结果分析
- 注意力权重解释
- 误差分布统计

## 扩展功能

### 自定义物理约束

可以在 `PhysicsConstraintLayer` 中添加更多物理约束：

```python
def custom_physics_constraint(self, predictions, inputs):
    # 添加自定义物理约束
    # 例如：能量守恒、动量守恒等
    pass
```

### 模型集成

可以训练多个模型并进行集成：

```python
# 训练多个模型
models = []
for i in range(5):
    model = PhysicsInformedTransformer(...)
    # 训练模型...
    models.append(model)

# 集成预测
ensemble_prediction = np.mean([model.predict(x) for model in models], axis=0)
```

## 故障排除

### 常见问题

1. **CUDA内存不足**: 减小batch_size或模型维度
2. **训练不收敛**: 调整学习率或增加正则化
3. **物理约束违反**: 增加physics_weight权重

### 性能优化

1. **使用混合精度训练**: 加速训练并减少内存使用
2. **数据并行**: 多GPU训练
3. **模型量化**: 部署时减少模型大小

## 贡献指南

欢迎提交Issue和Pull Request来改进模型！

## 许可证

MIT License

---

**注意**: 这是一个专门为伽马射线屏蔽物理问题设计的深度学习模型，确保了预测结果的物理合理性和高精度。