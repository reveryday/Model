import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from conf import *

class PhysicsConstraintLoss(nn.Module):
    """
    物理约束损失函数
    确保BUF预测结果满足以下物理定律：
    1. 累积因子 >= 1 (BUF_Cumulative >= 1)
    2. 有限几何 <= 无限几何 (BUF_Finite <= BUF_Infinite)
    3. 随厚度增加的单调性约束
    """
    
    def __init__(self, 
                 cumulative_weight=PHYSICS_CUMULATIVE_WEIGHT,
                 geometry_weight=PHYSICS_GEOMETRY_WEIGHT,
                 monotonicity_weight=PHYSICS_MONOTONICITY_WEIGHT,
                 min_buf_value=MIN_BUF_VALUE):
        super(PhysicsConstraintLoss, self).__init__()
        self.cumulative_weight = cumulative_weight
        self.geometry_weight = geometry_weight
        self.monotonicity_weight = monotonicity_weight
        self.min_buf_value = min_buf_value
        
        # 定义BUF列的索引（基于TARGET_COLUMNS）
        # TARGET_COLUMNS = ['Inf_Flu_BUF', 'Fin_Flu_BUF', 'Inf_Exp_BUF', 'Fin_Exp_BUF', 'Inf_Eff_BUF', 'Fin_Eff_BUF']
        self.buf_indices = {
            'inf_flu': 0,  # Inf_Flu_BUF
            'fin_flu': 1,  # Fin_Flu_BUF  
            'inf_exp': 2,  # Inf_Exp_BUF
            'fin_exp': 3,  # Fin_Exp_BUF
            'inf_eff': 4,  # Inf_Eff_BUF
            'fin_eff': 5   # Fin_Eff_BUF
        }
    
    def cumulative_constraint_loss(self, predictions):
        """
        累积因子约束：所有BUF >= 1
        对所有BUF类型应用此约束
        """
        total_violation = 0
        
        for buf_name, buf_idx in self.buf_indices.items():
            buf_values = predictions[:, buf_idx]
            
            # 将log1p变换后的预测转换回原始尺度
            buf_original = torch.expm1(buf_values)
            
            # 计算违反约束的程度（当BUF < min_value时）
            violation = F.relu(self.min_buf_value - buf_original)
            total_violation += torch.mean(violation ** 2)
        
        return total_violation
    
    def geometry_constraint_loss(self, predictions):
        """
        几何约束：有限几何 <= 无限几何
        Fin_* <= Inf_* (对于相同的物理量类型)
        """
        total_violation = 0
        
        # 对每种物理量类型应用约束：Fin <= Inf
        geometry_pairs = [
            ('fin_flu', 'inf_flu'),  # Fin_Flu_BUF <= Inf_Flu_BUF
            ('fin_exp', 'inf_exp'),  # Fin_Exp_BUF <= Inf_Exp_BUF  
            ('fin_eff', 'inf_eff')   # Fin_Eff_BUF <= Inf_Eff_BUF
        ]
        
        for fin_key, inf_key in geometry_pairs:
            fin_buf = predictions[:, self.buf_indices[fin_key]]
            inf_buf = predictions[:, self.buf_indices[inf_key]]
            
            # 将log1p变换后的预测转换回原始尺度
            fin_original = torch.expm1(fin_buf)
            inf_original = torch.expm1(inf_buf)
            
            # 计算违反约束的程度（当有限几何 > 无限几何时）
            violation = F.relu(fin_original - inf_original)
            total_violation += torch.mean(violation ** 2)
        
        return total_violation
    
    def monotonicity_constraint_loss(self, predictions, features, batch_indices=None):
        """
        单调性约束：随着厚度(MFP)增加，BUF值应该单调变化
        """
        if batch_indices is None or len(batch_indices) < 2:
            return torch.tensor(0.0, device=predictions.device)
        
        # 获取MFP值（假设是特征的第3列，索引2）
        mfp_values = features[:, 2]
        
        # 按MFP排序
        sorted_indices = torch.argsort(mfp_values)
        sorted_predictions = predictions[sorted_indices]
        sorted_mfp = mfp_values[sorted_indices]
        
        # 计算相邻点之间的差异
        mfp_diff = sorted_mfp[1:] - sorted_mfp[:-1]
        pred_diff = sorted_predictions[1:] - sorted_predictions[:-1]
        
        # 对于MFP增加的情况，BUF应该增加（或至少不显著减少）
        monotonicity_violations = []
        
        for buf_name, buf_idx in self.buf_indices.items():
            buf_diff = pred_diff[:, buf_idx]
            
            # 当MFP增加时，BUF不应该显著减少
            violation = F.relu(-buf_diff) * (mfp_diff > 0).float()
            monotonicity_violations.append(torch.mean(violation ** 2))
        
        monotonicity_loss = sum(monotonicity_violations)
        
        return monotonicity_loss
    
    def forward(self, predictions, features=None, batch_indices=None):
        """
        计算总的物理约束损失
        
        Args:
            predictions: [batch_size, output_dim] 模型预测（log1p变换后）
            features: [batch_size, input_dim] 输入特征
            batch_indices: 批次索引，用于单调性约束
        """
        total_loss = torch.tensor(0.0, device=predictions.device)
        loss_components = {}
        
        # 1. 累积因子约束
        if self.cumulative_weight > 0:
            cumulative_loss = self.cumulative_constraint_loss(predictions)
            total_loss += self.cumulative_weight * cumulative_loss
            loss_components['cumulative'] = cumulative_loss.item()
        
        # 2. 几何约束
        if self.geometry_weight > 0:
            geometry_loss = self.geometry_constraint_loss(predictions)
            total_loss += self.geometry_weight * geometry_loss
            loss_components['geometry'] = geometry_loss.item()
        
        # 3. 单调性约束
        if self.monotonicity_weight > 0 and features is not None:
            monotonicity_loss = self.monotonicity_constraint_loss(
                predictions, features, batch_indices
            )
            total_loss += self.monotonicity_weight * monotonicity_loss
            loss_components['monotonicity'] = monotonicity_loss.item()
        
        return total_loss, loss_components

class PhysicsInformedLoss(nn.Module):
    """
    物理信息损失函数，结合MSE损失和物理约束损失
    """
    
    def __init__(self, 
                 mse_weight=1.0,
                 physics_weight=PHYSICS_LOSS_WEIGHT,
                 cumulative_weight=PHYSICS_CUMULATIVE_WEIGHT,
                 geometry_weight=PHYSICS_GEOMETRY_WEIGHT,
                 monotonicity_weight=PHYSICS_MONOTONICITY_WEIGHT):
        super(PhysicsInformedLoss, self).__init__()
        
        self.mse_weight = mse_weight
        self.physics_weight = physics_weight
        
        # MSE损失
        self.mse_loss = nn.MSELoss()
        
        # 物理约束损失
        self.physics_constraint = PhysicsConstraintLoss(
            cumulative_weight=cumulative_weight,
            geometry_weight=geometry_weight,
            monotonicity_weight=monotonicity_weight
        )
    
    def forward(self, predictions, targets, features=None, batch_indices=None):
        """
        计算总损失
        
        Args:
            predictions: [batch_size, output_dim] 模型预测
            targets: [batch_size, output_dim] 真实标签
            features: [batch_size, input_dim] 输入特征
            batch_indices: 批次索引
        """
        # MSE损失
        mse_loss = self.mse_loss(predictions, targets)
        
        # 物理约束损失
        physics_loss, physics_components = self.physics_constraint(
            predictions, features, batch_indices
        )
        
        # 总损失
        total_loss = self.mse_weight * mse_loss + self.physics_weight * physics_loss
        
        # 返回损失组件用于监控
        loss_info = {
            'total_loss': total_loss.item(),
            'mse_loss': mse_loss.item(),
            'physics_loss': physics_loss.item(),
            **{f'physics_{k}': v for k, v in physics_components.items()}
        }
        
        return total_loss, loss_info

class AdaptivePhysicsLoss(nn.Module):
    """
    自适应物理约束损失，根据训练进度调整权重
    """
    
    def __init__(self, 
                 base_physics_weight=PHYSICS_LOSS_WEIGHT,
                 warmup_epochs=10,
                 max_epochs=100):
        super(AdaptivePhysicsLoss, self).__init__()
        
        self.base_physics_weight = base_physics_weight
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        
        self.physics_informed_loss = PhysicsInformedLoss(physics_weight=1.0)
        
    def get_physics_weight(self, epoch):
        """根据训练轮数计算物理约束权重"""
        if epoch < self.warmup_epochs:
            # 预热阶段：逐渐增加物理约束权重
            weight = self.base_physics_weight * (epoch / self.warmup_epochs)
        else:
            # 正常训练阶段：使用基础权重
            weight = self.base_physics_weight
        
        return weight
    
    def forward(self, predictions, targets, features=None, batch_indices=None, epoch=0):
        """
        计算自适应物理约束损失
        """
        # 获取当前轮次的物理约束权重
        current_physics_weight = self.get_physics_weight(epoch)
        
        # 更新物理约束权重
        self.physics_informed_loss.physics_weight = current_physics_weight
        
        return self.physics_informed_loss(predictions, targets, features, batch_indices)

def validate_physics_constraints(predictions, features=None, return_details=False):
    """
    验证预测结果是否满足物理约束
    
    Args:
        predictions: [batch_size, output_dim] 模型预测（log1p变换后）
        features: [batch_size, input_dim] 输入特征
        return_details: 是否返回详细的违反信息
    
    Returns:
        dict: 约束满足情况统计
    """
    device = predictions.device
    batch_size = predictions.size(0)
    
    # 转换回原始尺度
    predictions_original = torch.expm1(predictions)
    
    # 定义BUF列索引
    buf_indices = {
        'cumulative': TARGET_COLUMNS.index('BUF_Cumulative'),
        'finite': TARGET_COLUMNS.index('BUF_Finite'),
        'infinite': TARGET_COLUMNS.index('BUF_Infinite'),
        'point': TARGET_COLUMNS.index('BUF_Point'),
        'slab': TARGET_COLUMNS.index('BUF_Slab'),
        'sphere': TARGET_COLUMNS.index('BUF_Sphere')
    }
    
    validation_results = {}
    
    # 1. 累积因子约束验证
    cumulative_buf = predictions_original[:, buf_indices['cumulative']]
    cumulative_violations = (cumulative_buf < MIN_BUF_VALUE).sum().item()
    validation_results['cumulative_violations'] = cumulative_violations
    validation_results['cumulative_violation_rate'] = cumulative_violations / batch_size
    
    # 2. 几何约束验证
    finite_buf = predictions_original[:, buf_indices['finite']]
    infinite_buf = predictions_original[:, buf_indices['infinite']]
    point_buf = predictions_original[:, buf_indices['point']]
    slab_buf = predictions_original[:, buf_indices['slab']]
    sphere_buf = predictions_original[:, buf_indices['sphere']]
    
    geometry_violations = (
        (finite_buf > infinite_buf).sum() +
        (point_buf > infinite_buf).sum() +
        (slab_buf > infinite_buf).sum() +
        (sphere_buf > infinite_buf).sum()
    ).item()
    
    validation_results['geometry_violations'] = geometry_violations
    validation_results['geometry_violation_rate'] = geometry_violations / (batch_size * 4)
    
    # 3. 总体约束满足率
    total_constraints = batch_size * 5  # 1个累积约束 + 4个几何约束
    total_violations = cumulative_violations + geometry_violations
    validation_results['total_violation_rate'] = total_violations / total_constraints
    validation_results['constraint_satisfaction_rate'] = 1 - validation_results['total_violation_rate']
    
    if return_details:
        validation_results['details'] = {
            'cumulative_values': cumulative_buf.detach().cpu().numpy(),
            'finite_vs_infinite': (finite_buf - infinite_buf).detach().cpu().numpy(),
            'point_vs_infinite': (point_buf - infinite_buf).detach().cpu().numpy(),
            'slab_vs_infinite': (slab_buf - infinite_buf).detach().cpu().numpy(),
            'sphere_vs_infinite': (sphere_buf - infinite_buf).detach().cpu().numpy()
        }
    
    return validation_results

if __name__ == "__main__":
    # 测试物理约束损失函数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试数据
    batch_size = 32
    predictions = torch.randn(batch_size, OUTPUT_DIM, device=device)
    targets = torch.randn(batch_size, OUTPUT_DIM, device=device)
    features = torch.randn(batch_size, INPUT_DIM, device=device)
    
    # 测试物理约束损失
    physics_loss = PhysicsConstraintLoss().to(device)
    constraint_loss, components = physics_loss(predictions, features)
    
    print(f"物理约束损失测试:")
    print(f"总约束损失: {constraint_loss.item():.6f}")
    print(f"损失组件: {components}")
    
    # 测试物理信息损失
    physics_informed_loss = PhysicsInformedLoss().to(device)
    total_loss, loss_info = physics_informed_loss(predictions, targets, features)
    
    print(f"\n物理信息损失测试:")
    print(f"总损失: {total_loss.item():.6f}")
    print(f"损失信息: {loss_info}")
    
    # 测试约束验证
    validation_results = validate_physics_constraints(predictions, features, return_details=True)
    print(f"\n约束验证结果:")
    print(f"约束满足率: {validation_results['constraint_satisfaction_rate']:.3f}")
    print(f"累积约束违反率: {validation_results['cumulative_violation_rate']:.3f}")
    print(f"几何约束违反率: {validation_results['geometry_violation_rate']:.3f}")