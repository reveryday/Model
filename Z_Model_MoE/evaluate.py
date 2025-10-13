import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from conf import *
from utils import load_data, inverse_transform_predictions, get_shell_groups_stats
from model_moe import ShellMoEModel
from physics_loss import validate_physics_constraints

class ModelEvaluator:
    """模型评估器，支持分Shell统计分析"""
    
    def __init__(self, model_path, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = ShellMoEModel().to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"模型加载成功: {model_path}")
        print(f"最佳验证损失: {checkpoint.get('val_loss', 'N/A')}")
        print(f"训练轮数: {checkpoint.get('epoch', 'N/A')}")
    
    def evaluate_dataset(self, data_loader, dataset_name="Test"):
        """评估数据集"""
        print(f"\n评估 {dataset_name} 数据集...")
        
        all_predictions = []
        all_targets = []
        all_features = []
        all_shell_indices = []
        all_gate_weights = []
        
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for features, targets in tqdm(data_loader, desc=f"评估{dataset_name}"):
                features, targets = features.to(self.device), targets.to(self.device)
                batch_size = features.size(0)
                
                # 前向传播
                output = self.model(features)
                predictions = output['predictions']
                gate_weights = output['gate_weights']
                
                # 计算MSE损失
                mse_loss = nn.MSELoss()(predictions, targets)
                total_loss += mse_loss.item() * batch_size
                total_samples += batch_size
                
                # 收集结果
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
                all_features.append(features.cpu())
                all_gate_weights.append(gate_weights.cpu())
                
                # 提取Shell索引
                shell_indices = features[:, 1].long().cpu()  # Shell是第2列
                all_shell_indices.append(shell_indices)
        
        # 合并所有批次的结果
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        features = torch.cat(all_features, dim=0)
        shell_indices = torch.cat(all_shell_indices, dim=0)
        gate_weights = torch.cat(all_gate_weights, dim=0)
        
        avg_mse = total_loss / total_samples
        
        # 转换回原始尺度进行评估
        pred_original = inverse_transform_predictions(predictions)
        target_original = inverse_transform_predictions(targets)
        
        # 计算整体指标
        overall_metrics = self._calculate_metrics(pred_original, target_original)
        overall_metrics['mse_loss'] = avg_mse
        
        # 验证物理约束
        constraint_results = validate_physics_constraints(predictions, features, return_details=True)
        
        # 分Shell统计
        shell_stats = get_shell_groups_stats(
            pred_original, target_original, shell_indices.numpy(), SHELL_GROUPS
        )
        
        # 专家使用分析
        expert_analysis = self._analyze_expert_usage(gate_weights, shell_indices)
        
        return {
            'overall_metrics': overall_metrics,
            'constraint_results': constraint_results,
            'shell_stats': shell_stats,
            'expert_analysis': expert_analysis,
            'predictions': pred_original,
            'targets': target_original,
            'shell_indices': shell_indices.numpy(),
            'gate_weights': gate_weights.numpy()
        }
    
    def _calculate_metrics(self, predictions, targets):
        """计算评估指标"""
        metrics = {}
        
        for i, col_name in enumerate(TARGET_COLUMNS):
            pred_col = predictions[:, i]
            target_col = targets[:, i]
            
            # 过滤无效值
            valid_mask = ~(np.isnan(pred_col) | np.isnan(target_col) | 
                          np.isinf(pred_col) | np.isinf(target_col))
            
            if valid_mask.sum() > 0:
                pred_valid = pred_col[valid_mask]
                target_valid = target_col[valid_mask]
                
                metrics[col_name] = {
                    'mse': mean_squared_error(target_valid, pred_valid),
                    'mae': mean_absolute_error(target_valid, pred_valid),
                    'r2': r2_score(target_valid, pred_valid),
                    'mape': np.mean(np.abs((target_valid - pred_valid) / target_valid)) * 100
                }
            else:
                metrics[col_name] = {'mse': np.nan, 'mae': np.nan, 'r2': np.nan, 'mape': np.nan}
        
        # 计算平均指标
        valid_metrics = [m for m in metrics.values() if not np.isnan(m['mse'])]
        if valid_metrics:
            metrics['average'] = {
                'mse': np.mean([m['mse'] for m in valid_metrics]),
                'mae': np.mean([m['mae'] for m in valid_metrics]),
                'r2': np.mean([m['r2'] for m in valid_metrics]),
                'mape': np.mean([m['mape'] for m in valid_metrics])
            }
        
        return metrics
    
    def _analyze_expert_usage(self, gate_weights, shell_indices):
        """分析专家使用情况"""
        expert_analysis = {}
        
        # 整体专家使用分布
        overall_usage = np.mean(gate_weights, axis=0)
        expert_analysis['overall_usage'] = overall_usage.tolist()
        
        # 按Shell分组的专家使用
        shell_expert_usage = {}
        for i, (start_shell, end_shell) in enumerate(SHELL_GROUPS):
            mask = (shell_indices >= start_shell) & (shell_indices <= end_shell)
            if mask.sum() > 0:
                group_usage = np.mean(gate_weights[mask], axis=0)
                shell_expert_usage[f'Shell_{start_shell}-{end_shell}'] = group_usage.tolist()
        
        expert_analysis['shell_usage'] = shell_expert_usage
        
        # 专家专业化程度（熵）
        expert_entropy = []
        for i in range(gate_weights.shape[1]):
            expert_weights = gate_weights[:, i]
            # 计算该专家在不同样本上的权重分布熵
            if expert_weights.std() > 0:
                normalized_weights = (expert_weights - expert_weights.min()) / (expert_weights.max() - expert_weights.min() + 1e-8)
                entropy = -np.sum(normalized_weights * np.log(normalized_weights + 1e-8))
                expert_entropy.append(entropy)
            else:
                expert_entropy.append(0)
        
        expert_analysis['expert_entropy'] = expert_entropy
        
        return expert_analysis
    
    def plot_evaluation_results(self, eval_results, save_dir='./evaluation_results'):
        """绘制评估结果"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. 整体性能对比图
        self._plot_overall_performance(eval_results, save_dir)
        
        # 2. 分Shell性能分析
        self._plot_shell_performance(eval_results, save_dir)
        
        # 3. 预测vs真实值散点图
        self._plot_prediction_scatter(eval_results, save_dir)
        
        # 4. 专家使用分析
        self._plot_expert_analysis(eval_results, save_dir)
        
        # 5. 物理约束验证
        self._plot_constraint_validation(eval_results, save_dir)
        
        print(f"评估图表保存在: {save_dir}")
    
    def _plot_overall_performance(self, eval_results, save_dir):
        """绘制整体性能图"""
        metrics = eval_results['overall_metrics']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # MSE
        mse_values = [metrics[col]['mse'] for col in TARGET_COLUMNS if not np.isnan(metrics[col]['mse'])]
        axes[0, 0].bar(range(len(mse_values)), mse_values, alpha=0.7)
        axes[0, 0].set_title('MSE by Target Variable')
        axes[0, 0].set_xlabel('Target Index')
        axes[0, 0].set_ylabel('MSE')
        axes[0, 0].grid(True, alpha=0.3)
        
        # MAE
        mae_values = [metrics[col]['mae'] for col in TARGET_COLUMNS if not np.isnan(metrics[col]['mae'])]
        axes[0, 1].bar(range(len(mae_values)), mae_values, alpha=0.7, color='orange')
        axes[0, 1].set_title('MAE by Target Variable')
        axes[0, 1].set_xlabel('Target Index')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].grid(True, alpha=0.3)
        
        # R²
        r2_values = [metrics[col]['r2'] for col in TARGET_COLUMNS if not np.isnan(metrics[col]['r2'])]
        axes[1, 0].bar(range(len(r2_values)), r2_values, alpha=0.7, color='green')
        axes[1, 0].set_title('R² by Target Variable')
        axes[1, 0].set_xlabel('Target Index')
        axes[1, 0].set_ylabel('R²')
        axes[1, 0].grid(True, alpha=0.3)
        
        # MAPE
        mape_values = [metrics[col]['mape'] for col in TARGET_COLUMNS if not np.isnan(metrics[col]['mape'])]
        axes[1, 1].bar(range(len(mape_values)), mape_values, alpha=0.7, color='red')
        axes[1, 1].set_title('MAPE by Target Variable')
        axes[1, 1].set_xlabel('Target Index')
        axes[1, 1].set_ylabel('MAPE (%)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'overall_performance.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_shell_performance(self, eval_results, save_dir):
        """绘制分Shell性能分析"""
        shell_stats = eval_results['shell_stats']
        
        if not shell_stats:
            return
        
        shell_names = list(shell_stats.keys())
        metrics_names = ['mse', 'mae', 'r2', 'mape']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics_names):
            values = [shell_stats[shell][metric] for shell in shell_names]
            
            bars = axes[i].bar(range(len(values)), values, alpha=0.7)
            axes[i].set_title(f'{metric.upper()} by Shell Group')
            axes[i].set_xlabel('Shell Group')
            axes[i].set_ylabel(metric.upper())
            axes[i].set_xticks(range(len(shell_names)))
            axes[i].set_xticklabels(shell_names, rotation=45)
            axes[i].grid(True, alpha=0.3)
            
            # 添加数值标签
            for j, bar in enumerate(bars):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.4f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'shell_performance.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_prediction_scatter(self, eval_results, save_dir):
        """绘制预测vs真实值散点图"""
        predictions = eval_results['predictions']
        targets = eval_results['targets']
        
        n_targets = len(TARGET_COLUMNS)
        cols = 3
        rows = (n_targets + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, col_name in enumerate(TARGET_COLUMNS):
            row, col = i // cols, i % cols
            
            pred_col = predictions[:, i]
            target_col = targets[:, i]
            
            # 过滤无效值
            valid_mask = ~(np.isnan(pred_col) | np.isnan(target_col) | 
                          np.isinf(pred_col) | np.isinf(target_col))
            
            if valid_mask.sum() > 0:
                pred_valid = pred_col[valid_mask]
                target_valid = target_col[valid_mask]
                
                axes[row, col].scatter(target_valid, pred_valid, alpha=0.5, s=1)
                
                # 添加y=x线
                min_val = min(target_valid.min(), pred_valid.min())
                max_val = max(target_valid.max(), pred_valid.max())
                axes[row, col].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
                
                # 计算R²
                r2 = r2_score(target_valid, pred_valid)
                axes[row, col].set_title(f'{col_name} (R²={r2:.3f})')
                axes[row, col].set_xlabel('True Values')
                axes[row, col].set_ylabel('Predictions')
                axes[row, col].grid(True, alpha=0.3)
            else:
                axes[row, col].text(0.5, 0.5, 'No valid data', ha='center', va='center')
                axes[row, col].set_title(f'{col_name} (No data)')
        
        # 隐藏多余的子图
        for i in range(n_targets, rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'prediction_scatter.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_expert_analysis(self, eval_results, save_dir):
        """绘制专家使用分析"""
        expert_analysis = eval_results['expert_analysis']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 整体专家使用分布
        overall_usage = expert_analysis['overall_usage']
        axes[0, 0].bar(range(len(overall_usage)), overall_usage, alpha=0.7)
        axes[0, 0].set_title('Overall Expert Usage Distribution')
        axes[0, 0].set_xlabel('Expert Index')
        axes[0, 0].set_ylabel('Usage Rate')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 专家专业化程度
        expert_entropy = expert_analysis['expert_entropy']
        axes[0, 1].bar(range(len(expert_entropy)), expert_entropy, alpha=0.7, color='orange')
        axes[0, 1].set_title('Expert Specialization (Entropy)')
        axes[0, 1].set_xlabel('Expert Index')
        axes[0, 1].set_ylabel('Entropy')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 按Shell分组的专家使用热力图
        shell_usage = expert_analysis['shell_usage']
        if shell_usage:
            shell_names = list(shell_usage.keys())
            usage_matrix = np.array([shell_usage[shell] for shell in shell_names])
            
            im = axes[1, 0].imshow(usage_matrix, cmap='viridis', aspect='auto')
            axes[1, 0].set_title('Expert Usage by Shell Group')
            axes[1, 0].set_xlabel('Expert Index')
            axes[1, 0].set_ylabel('Shell Group')
            axes[1, 0].set_yticks(range(len(shell_names)))
            axes[1, 0].set_yticklabels(shell_names)
            plt.colorbar(im, ax=axes[1, 0])
        
        # 专家负载均衡
        usage_std = np.std(overall_usage)
        usage_mean = np.mean(overall_usage)
        axes[1, 1].axhline(y=usage_mean, color='r', linestyle='--', label=f'Mean: {usage_mean:.3f}')
        axes[1, 1].fill_between(range(len(overall_usage)), 
                               usage_mean - usage_std, usage_mean + usage_std, 
                               alpha=0.3, label=f'±1 STD: {usage_std:.3f}')
        axes[1, 1].bar(range(len(overall_usage)), overall_usage, alpha=0.7)
        axes[1, 1].set_title('Expert Load Balancing')
        axes[1, 1].set_xlabel('Expert Index')
        axes[1, 1].set_ylabel('Usage Rate')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'expert_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_constraint_validation(self, eval_results, save_dir):
        """绘制物理约束验证结果"""
        constraint_results = eval_results['constraint_results']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 约束满足率
        satisfaction_rate = constraint_results['constraint_satisfaction_rate']
        cumulative_rate = 1 - constraint_results['cumulative_violation_rate']
        geometry_rate = 1 - constraint_results['geometry_violation_rate']
        
        rates = [satisfaction_rate, cumulative_rate, geometry_rate]
        labels = ['Overall', 'Cumulative', 'Geometry']
        colors = ['green', 'blue', 'orange']
        
        bars = axes[0, 0].bar(labels, rates, color=colors, alpha=0.7)
        axes[0, 0].set_title('Constraint Satisfaction Rates')
        axes[0, 0].set_ylabel('Satisfaction Rate')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, rate in zip(bars, rates):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{rate:.3f}', ha='center', va='bottom')
        
        # 如果有详细信息，绘制约束违反分布
        if 'details' in constraint_results:
            details = constraint_results['details']
            
            # 累积因子分布
            cumulative_values = details['cumulative_values']
            axes[0, 1].hist(cumulative_values, bins=50, alpha=0.7, color='blue')
            axes[0, 1].axvline(x=1, color='r', linestyle='--', label='Min BUF = 1')
            axes[0, 1].set_title('Cumulative BUF Distribution')
            axes[0, 1].set_xlabel('BUF Value')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # 几何约束违反分布
            geometry_diffs = [
                details['finite_vs_infinite'],
                details['point_vs_infinite'],
                details['slab_vs_infinite'],
                details['sphere_vs_infinite']
            ]
            geometry_labels = ['Finite-Infinite', 'Point-Infinite', 'Slab-Infinite', 'Sphere-Infinite']
            
            for i, (diff, label) in enumerate(zip(geometry_diffs, geometry_labels)):
                axes[0, 2].hist(diff, bins=30, alpha=0.5, label=label)
            
            axes[0, 2].axvline(x=0, color='r', linestyle='--', label='Constraint Boundary')
            axes[0, 2].set_title('Geometry Constraint Violations')
            axes[0, 2].set_xlabel('Difference (Finite - Infinite)')
            axes[0, 2].set_ylabel('Frequency')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # 违反统计
        violation_counts = [
            constraint_results['cumulative_violations'],
            constraint_results['geometry_violations']
        ]
        violation_labels = ['Cumulative', 'Geometry']
        
        axes[1, 0].bar(violation_labels, violation_counts, alpha=0.7, color=['red', 'orange'])
        axes[1, 0].set_title('Constraint Violations Count')
        axes[1, 0].set_ylabel('Number of Violations')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 约束满足率趋势（如果有多个评估点）
        axes[1, 1].text(0.5, 0.5, 'Constraint Satisfaction\nTrend Analysis\n(Requires multiple evaluations)', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Satisfaction Trend')
        
        # 物理合理性评分
        physics_score = satisfaction_rate * 100
        axes[1, 2].pie([physics_score, 100-physics_score], 
                      labels=[f'Satisfied ({physics_score:.1f}%)', f'Violated ({100-physics_score:.1f}%)'],
                      colors=['green', 'red'], autopct='%1.1f%%')
        axes[1, 2].set_title('Physics Constraint Score')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'constraint_validation.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_evaluation_report(self, eval_results, save_dir='./evaluation_results'):
        """生成评估报告"""
        os.makedirs(save_dir, exist_ok=True)
        
        report = {
            'evaluation_summary': {
                'overall_metrics': eval_results['overall_metrics'],
                'constraint_satisfaction_rate': eval_results['constraint_results']['constraint_satisfaction_rate'],
                'shell_performance': eval_results['shell_stats'],
                'expert_usage': eval_results['expert_analysis']['overall_usage']
            },
            'detailed_results': eval_results
        }
        
        # 保存JSON报告
        with open(os.path.join(save_dir, 'evaluation_report.json'), 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        # 生成文本报告
        self._generate_text_report(eval_results, save_dir)
        
        print(f"评估报告保存在: {save_dir}")
    
    def _generate_text_report(self, eval_results, save_dir):
        """生成文本格式的评估报告"""
        with open(os.path.join(save_dir, 'evaluation_report.txt'), 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("Shell MoE 模型评估报告\n")
            f.write("=" * 60 + "\n\n")
            
            # 整体性能
            f.write("1. 整体性能指标\n")
            f.write("-" * 30 + "\n")
            overall_metrics = eval_results['overall_metrics']
            if 'average' in overall_metrics:
                avg_metrics = overall_metrics['average']
                f.write(f"平均 MSE: {avg_metrics['mse']:.6f}\n")
                f.write(f"平均 MAE: {avg_metrics['mae']:.6f}\n")
                f.write(f"平均 R²: {avg_metrics['r2']:.6f}\n")
                f.write(f"平均 MAPE: {avg_metrics['mape']:.2f}%\n\n")
            
            # 物理约束满足情况
            f.write("2. 物理约束满足情况\n")
            f.write("-" * 30 + "\n")
            constraint_results = eval_results['constraint_results']
            f.write(f"总体约束满足率: {constraint_results['constraint_satisfaction_rate']:.3f}\n")
            f.write(f"累积因子约束满足率: {1-constraint_results['cumulative_violation_rate']:.3f}\n")
            f.write(f"几何约束满足率: {1-constraint_results['geometry_violation_rate']:.3f}\n\n")
            
            # 分Shell性能
            f.write("3. 分Shell性能分析\n")
            f.write("-" * 30 + "\n")
            shell_stats = eval_results['shell_stats']
            for shell_name, stats in shell_stats.items():
                f.write(f"{shell_name}:\n")
                f.write(f"  样本数: {stats['count']}\n")
                f.write(f"  MSE: {stats['mse']:.6f}\n")
                f.write(f"  MAE: {stats['mae']:.6f}\n")
                f.write(f"  R²: {stats['r2']:.6f}\n")
                f.write(f"  MAPE: {stats['mape']:.2f}%\n\n")
            
            # 专家使用分析
            f.write("4. 专家使用分析\n")
            f.write("-" * 30 + "\n")
            expert_analysis = eval_results['expert_analysis']
            overall_usage = expert_analysis['overall_usage']
            f.write("专家使用分布:\n")
            for i, usage in enumerate(overall_usage):
                f.write(f"  专家 {i}: {usage:.3f}\n")
            
            usage_std = np.std(overall_usage)
            f.write(f"\n负载均衡度 (标准差): {usage_std:.3f}\n")
            f.write(f"专家使用均匀性: {'良好' if usage_std < 0.1 else '需要改进'}\n")

def main():
    """主评估函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='评估Shell MoE模型')
    parser.add_argument('--model_path', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--data_path', type=str, default=DATA_PATH, help='数据文件路径')
    parser.add_argument('--save_dir', type=str, default='./evaluation_results', help='结果保存目录')
    parser.add_argument('--device', type=str, default=None, help='计算设备')
    
    args = parser.parse_args()
    
    # 设置设备
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    print("加载数据...")
    data_dict = load_data(
        data_path=args.data_path,
        val_size=VAL_SIZE,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        batch_size=BATCH_SIZE,
        use_weighted_sampling=False  # 评估时不使用加权采样
    )
    
    # 创建评估器
    evaluator = ModelEvaluator(args.model_path, device)
    
    # 评估测试集
    test_results = evaluator.evaluate_dataset(data_dict['test_loader'], "Test")
    
    # 绘制结果
    evaluator.plot_evaluation_results(test_results, args.save_dir)
    
    # 生成报告
    evaluator.generate_evaluation_report(test_results, args.save_dir)
    
    print("评估完成！")

if __name__ == "__main__":
    main()