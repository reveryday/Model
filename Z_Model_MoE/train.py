import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from conf import *
from utils import load_data, AddNoise, inverse_transform_predictions
from model_moe import ShellMoEModel
from physics_loss import PhysicsInformedLoss, AdaptivePhysicsLoss, validate_physics_constraints

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=EARLY_STOPPING_PATIENCE, min_delta=1e-6, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, model, device, use_adaptive_physics=True):
        self.model = model.to(device)
        self.device = device
        self.use_adaptive_physics = use_adaptive_physics
        
        # 损失函数
        if use_adaptive_physics:
            self.criterion = AdaptivePhysicsLoss(
                base_physics_weight=PHYSICS_LOSS_WEIGHT,
                warmup_epochs=10,
                max_epochs=NUM_EPOCHS
            ).to(device)
        else:
            self.criterion = PhysicsInformedLoss().to(device)
        
        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # 学习率调度器
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=T_0,
            T_mult=T_MULT,
            eta_min=LEARNING_RATE * 0.01
        )
        
        # 早停
        self.early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mse': [],
            'val_mse': [],
            'train_physics': [],
            'val_physics': [],
            'learning_rate': [],
            'expert_usage': [],
            'constraint_satisfaction': []
        }
        
    def train_epoch(self, train_loader, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_mse = 0
        total_physics = 0
        total_samples = 0
        
        # 专家使用统计
        epoch_expert_usage = []
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}')
        
        for batch_idx, (features, targets) in enumerate(progress_bar):
            features, targets = features.to(self.device), targets.to(self.device)
            batch_size = features.size(0)
            
            # 添加噪声增强
            if NOISE_LEVEL > 0:
                features = AddNoise(features, NOISE_LEVEL)
            
            # 前向传播
            self.optimizer.zero_grad()
            
            output = self.model(features)
            predictions = output['predictions']
            gate_weights = output['gate_weights']
            
            # 计算损失
            if self.use_adaptive_physics:
                loss, loss_info = self.criterion(predictions, targets, features, epoch=epoch)
            else:
                loss, loss_info = self.criterion(predictions, targets, features)
            
            # 添加负载均衡损失
            load_balancing_loss = self.model.compute_load_balancing_loss(gate_weights)
            loss += load_balancing_loss
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            if GRADIENT_CLIP_VALUE > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRADIENT_CLIP_VALUE)
            
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item() * batch_size
            total_mse += loss_info['mse_loss'] * batch_size
            total_physics += loss_info['physics_loss'] * batch_size
            total_samples += batch_size
            
            # 专家使用统计
            expert_usage = self.model.get_expert_usage(gate_weights)
            epoch_expert_usage.append(expert_usage.detach().cpu().numpy())
            
            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'MSE': f'{loss_info["mse_loss"]:.6f}',
                'Physics': f'{loss_info["physics_loss"]:.6f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        # 计算平均值
        avg_loss = total_loss / total_samples
        avg_mse = total_mse / total_samples
        avg_physics = total_physics / total_samples
        avg_expert_usage = np.mean(epoch_expert_usage, axis=0)
        
        return avg_loss, avg_mse, avg_physics, avg_expert_usage
    
    def validate_epoch(self, val_loader, epoch):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0
        total_mse = 0
        total_physics = 0
        total_samples = 0
        
        all_predictions = []
        all_targets = []
        all_features = []
        
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(self.device), targets.to(self.device)
                batch_size = features.size(0)
                
                # 前向传播
                output = self.model(features)
                predictions = output['predictions']
                
                # 计算损失
                if self.use_adaptive_physics:
                    loss, loss_info = self.criterion(predictions, targets, features, epoch=epoch)
                else:
                    loss, loss_info = self.criterion(predictions, targets, features)
                
                # 统计
                total_loss += loss.item() * batch_size
                total_mse += loss_info['mse_loss'] * batch_size
                total_physics += loss_info['physics_loss'] * batch_size
                total_samples += batch_size
                
                # 收集预测结果用于约束验证
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
                all_features.append(features.cpu())
        
        # 计算平均值
        avg_loss = total_loss / total_samples
        avg_mse = total_mse / total_samples
        avg_physics = total_physics / total_samples
        
        # 验证物理约束
        all_predictions = torch.cat(all_predictions, dim=0)
        all_features = torch.cat(all_features, dim=0)
        constraint_results = validate_physics_constraints(all_predictions, all_features)
        
        return avg_loss, avg_mse, avg_physics, constraint_results
    
    def train(self, train_loader, val_loader, save_dir='./checkpoints'):
        """完整训练流程"""
        print(f"开始训练 Shell MoE 模型")
        print(f"设备: {self.device}")
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"训练样本数: {len(train_loader.dataset)}")
        print(f"验证样本数: {len(val_loader.dataset)}")
        print("-" * 50)
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_loss = float('inf')
        start_time = time.time()
        
        for epoch in range(NUM_EPOCHS):
            epoch_start_time = time.time()
            
            # 训练
            train_loss, train_mse, train_physics, expert_usage = self.train_epoch(train_loader, epoch)
            
            # 验证
            val_loss, val_mse, val_physics, constraint_results = self.validate_epoch(val_loader, epoch)
            
            # 学习率调度
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_mse'].append(train_mse)
            self.history['val_mse'].append(val_mse)
            self.history['train_physics'].append(train_physics)
            self.history['val_physics'].append(val_physics)
            self.history['learning_rate'].append(current_lr)
            self.history['expert_usage'].append(expert_usage.tolist())
            self.history['constraint_satisfaction'].append(constraint_results['constraint_satisfaction_rate'])
            
            # 打印进度
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} ({epoch_time:.1f}s)")
            print(f"  Train - Loss: {train_loss:.6f}, MSE: {train_mse:.6f}, Physics: {train_physics:.6f}")
            print(f"  Val   - Loss: {val_loss:.6f}, MSE: {val_mse:.6f}, Physics: {val_physics:.6f}")
            print(f"  约束满足率: {constraint_results['constraint_satisfaction_rate']:.3f}")
            print(f"  专家使用分布: {expert_usage}")
            print(f"  学习率: {current_lr:.2e}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_loss': val_loss,
                    'history': self.history
                }, os.path.join(save_dir, 'best_model.pth'))
                print(f"  ✓ 保存最佳模型 (val_loss: {val_loss:.6f})")
            
            # 早停检查
            if self.early_stopping(val_loss, self.model):
                print(f"早停触发，在第 {epoch+1} 轮停止训练")
                break
            
            print("-" * 50)
        
        total_time = time.time() - start_time
        print(f"训练完成！总用时: {total_time/3600:.2f} 小时")
        
        # 保存训练历史
        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump(self.history, f, indent=2)
        
        return self.history
    
    def plot_training_history(self, save_dir='./checkpoints'):
        """绘制训练历史"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 损失曲线
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss', alpha=0.8)
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss', alpha=0.8)
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # MSE损失
        axes[0, 1].plot(self.history['train_mse'], label='Train MSE', alpha=0.8)
        axes[0, 1].plot(self.history['val_mse'], label='Val MSE', alpha=0.8)
        axes[0, 1].set_title('MSE Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MSE')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 物理约束损失
        axes[0, 2].plot(self.history['train_physics'], label='Train Physics', alpha=0.8)
        axes[0, 2].plot(self.history['val_physics'], label='Val Physics', alpha=0.8)
        axes[0, 2].set_title('Physics Constraint Loss')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Physics Loss')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 学习率
        axes[1, 0].plot(self.history['learning_rate'], alpha=0.8)
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('LR')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 约束满足率
        axes[1, 1].plot(self.history['constraint_satisfaction'], alpha=0.8)
        axes[1, 1].set_title('Constraint Satisfaction Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Satisfaction Rate')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 专家使用分布（最后一个epoch）
        if self.history['expert_usage']:
            final_expert_usage = self.history['expert_usage'][-1]
            axes[1, 2].bar(range(len(final_expert_usage)), final_expert_usage, alpha=0.8)
            axes[1, 2].set_title('Final Expert Usage Distribution')
            axes[1, 2].set_xlabel('Expert Index')
            axes[1, 2].set_ylabel('Usage Rate')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """主训练函数"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 设置随机种子
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_STATE)
    
    # 加载数据
    print("加载数据...")
    data_dict = load_data(
        data_path=DATA_PATH,
        val_size=VAL_SIZE,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        batch_size=BATCH_SIZE,
        use_weighted_sampling=True
    )
    
    train_loader = data_dict['train_loader']
    val_loader = data_dict['val_loader']
    test_loader = data_dict['test_loader']
    
    # 创建模型
    print("创建模型...")
    model = ShellMoEModel()
    
    # 创建训练器
    trainer = ModelTrainer(model, device, use_adaptive_physics=True)
    
    # 创建保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f'./checkpoints/shell_moe_{timestamp}'
    
    # 开始训练
    history = trainer.train(train_loader, val_loader, save_dir)
    
    # 绘制训练历史
    trainer.plot_training_history(save_dir)
    
    print(f"训练完成！模型和结果保存在: {save_dir}")

if __name__ == "__main__":
    main()