import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from physics_informed_transformer import PhysicsInformedTransformer, PhysicsInformedLoss

class BUFDataset(Dataset):
    """伽马射线屏蔽累积因子数据集"""
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        
        # 分离输入和输出特征
        self.input_features = self.data.iloc[:, :8].values.astype(np.float32)
        self.output_features = self.data.iloc[:, 8:].values.astype(np.float32)
        
        # 数据预处理
        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()  # 使用标准化，保持原始分布特性
        
        self.input_features = self.input_scaler.fit_transform(self.input_features)
        self.output_features = self.output_scaler.fit_transform(self.output_features)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_data = torch.tensor(self.input_features[idx], dtype=torch.float32)
        output_data = torch.tensor(self.output_features[idx], dtype=torch.float32)
        
        if self.transform:
            input_data = self.transform(input_data)
            
        return input_data, output_data
    
    def get_scalers(self):
        return self.input_scaler, self.output_scaler

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=10, min_delta=1e-6, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

class ModelTrainer:
    """模型训练器"""
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.train_physics_losses = []
        self.val_physics_losses = []
        
    def train_epoch(self, train_loader, optimizer, criterion):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_mse_loss = 0.0
        total_physics_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            
            # 前向传播
            outputs, _ = self.model(inputs)
            
            # 计算损失
            loss, mse_loss, physics_loss = criterion(outputs, targets, inputs)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            total_mse_loss += mse_loss.item()
            total_physics_loss += physics_loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'MSE': f'{mse_loss.item():.6f}',
                'Physics': f'{physics_loss.item():.6f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        avg_mse_loss = total_mse_loss / len(train_loader)
        avg_physics_loss = total_physics_loss / len(train_loader)
        
        return avg_loss, avg_mse_loss, avg_physics_loss
    
    def validate_epoch(self, val_loader, criterion):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        total_mse_loss = 0.0
        total_physics_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs, _ = self.model(inputs)
                loss, mse_loss, physics_loss = criterion(outputs, targets, inputs)
                
                total_loss += loss.item()
                total_mse_loss += mse_loss.item()
                total_physics_loss += physics_loss.item()
        
        avg_loss = total_loss / len(val_loader)
        avg_mse_loss = total_mse_loss / len(val_loader)
        avg_physics_loss = total_physics_loss / len(val_loader)
        
        return avg_loss, avg_mse_loss, avg_physics_loss
    
    def train(self, train_loader, val_loader, num_epochs=100, learning_rate=1e-4, 
              weight_decay=1e-5, patience=15):
        """完整训练过程"""
        
        # 优化器和学习率调度器
        optimizer = optim.AdamW(self.model.parameters(), 
                               lr=learning_rate, 
                               weight_decay=weight_decay)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # 损失函数
        criterion = PhysicsInformedLoss(mse_weight=1.0, physics_weight=0.1)
        
        # 早停
        early_stopping = EarlyStopping(patience=patience)
        
        print(f"开始训练，设备: {self.device}")
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # 训练
            train_loss, train_mse, train_physics = self.train_epoch(
                train_loader, optimizer, criterion
            )
            
            # 验证
            val_loss, val_mse, val_physics = self.validate_epoch(
                val_loader, criterion
            )
            
            # 记录损失
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_physics_losses.append(train_physics)
            self.val_physics_losses.append(val_physics)
            
            # 学习率调度
            scheduler.step(val_loss)
            
            print(f"训练损失: {train_loss:.6f} (MSE: {train_mse:.6f}, Physics: {train_physics:.6f})")
            print(f"验证损失: {val_loss:.6f} (MSE: {val_mse:.6f}, Physics: {val_physics:.6f})")
            print(f"当前学习率: {optimizer.param_groups[0]['lr']:.2e}")
            
            # 早停检查
            if early_stopping(val_loss, self.model):
                print(f"\n早停触发，在第 {epoch+1} 轮停止训练")
                break
        
        print("\n训练完成！")
        return self.train_losses, self.val_losses
    
    def evaluate(self, test_loader, output_scaler):
        """评估模型性能"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs, _ = self.model(inputs)
                
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        predictions = np.vstack(all_predictions)
        targets = np.vstack(all_targets)
        
        # 反归一化
        predictions = output_scaler.inverse_transform(predictions)
        targets = output_scaler.inverse_transform(targets)
        
        # 计算评估指标
        metrics = {}
        feature_names = ['Inf_Flu_BUF', 'Fin_Flu_BUF', 'Inf_Exp_BUF', 
                        'Fin_Exp_BUF', 'Inf_Eff_BUF', 'Fin_Eff_BUF']
        
        for i, name in enumerate(feature_names):
            mse = mean_squared_error(targets[:, i], predictions[:, i])
            mae = mean_absolute_error(targets[:, i], predictions[:, i])
            r2 = r2_score(targets[:, i], predictions[:, i])
            
            metrics[name] = {
                'MSE': mse,
                'RMSE': np.sqrt(mse),
                'MAE': mae,
                'R2': r2
            }
        
        return metrics, predictions, targets

def create_data_loaders(csv_file, batch_size=64, train_ratio=0.7, val_ratio=0.15):
    """创建数据加载器"""
    dataset = BUFDataset(csv_file)
    
    # 计算数据集大小
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    # 随机分割数据集
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader, dataset.get_scalers()

def plot_training_history(train_losses, val_losses, train_physics_losses, val_physics_losses):
    """绘制训练历史"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 总损失
    axes[0].plot(train_losses, label='训练损失', color='blue')
    axes[0].plot(val_losses, label='验证损失', color='red')
    axes[0].set_title('训练和验证损失')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # 物理损失
    axes[1].plot(train_physics_losses, label='训练物理损失', color='green')
    axes[1].plot(val_physics_losses, label='验证物理损失', color='orange')
    axes[1].set_title('物理约束损失')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Physics Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_predictions(predictions, targets, feature_names):
    """绘制预测结果对比"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, name in enumerate(feature_names):
        axes[i].scatter(targets[:, i], predictions[:, i], alpha=0.5, s=1)
        axes[i].plot([targets[:, i].min(), targets[:, i].max()], 
                    [targets[:, i].min(), targets[:, i].max()], 'r--', lw=2)
        axes[i].set_xlabel('真实值')
        axes[i].set_ylabel('预测值')
        axes[i].set_title(f'{name}')
        axes[i].grid(True)
        
        # 计算R²
        r2 = r2_score(targets[:, i], predictions[:, i])
        axes[i].text(0.05, 0.95, f'R² = {r2:.4f}', 
                    transform=axes[i].transAxes, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('prediction_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建数据加载器
    print("加载数据...")
    train_loader, val_loader, test_loader, (input_scaler, output_scaler) = create_data_loaders(
        'dataset.csv', batch_size=128
    )
    
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"验证集大小: {len(val_loader.dataset)}")
    print(f"测试集大小: {len(test_loader.dataset)}")
    
    # 创建模型
    print("\n创建模型...")
    model = PhysicsInformedTransformer(
        input_dim=8,
        output_dim=6,
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
        dropout=0.1
    )
    
    # 创建训练器
    trainer = ModelTrainer(model)
    
    # 训练模型
    print("\n开始训练...")
    train_losses, val_losses = trainer.train(
        train_loader, val_loader,
        num_epochs=10000,
        learning_rate=1e-4,
        weight_decay=1e-5,
        patience=15
    )
    
    # 绘制训练历史
    plot_training_history(train_losses, val_losses, 
                         trainer.train_physics_losses, trainer.val_physics_losses)
    
    # 评估模型
    print("\n评估模型...")
    metrics, predictions, targets = trainer.evaluate(test_loader, output_scaler)
    
    # 打印评估结果
    print("\n=== 模型评估结果 ===")
    for feature_name, metric in metrics.items():
        print(f"\n{feature_name}:")
        print(f"  RMSE: {metric['RMSE']:.6f}")
        print(f"  MAE:  {metric['MAE']:.6f}")
        print(f"  R²:   {metric['R2']:.6f}")
    
    # 绘制预测结果
    feature_names = ['Inf_Flu_BUF', 'Fin_Flu_BUF', 'Inf_Exp_BUF', 
                    'Fin_Exp_BUF', 'Inf_Eff_BUF', 'Fin_Eff_BUF']
    plot_predictions(predictions, targets, feature_names)
    
    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_scaler': input_scaler,
        'output_scaler': output_scaler,
        'metrics': metrics
    }, 'physics_informed_transformer_model.pth')
    
    print("\n模型已保存为 'physics_informed_transformer_model.pth'")