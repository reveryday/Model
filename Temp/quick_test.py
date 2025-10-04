import torch
import numpy as np
from training_pipeline import create_data_loaders, ModelTrainer
from physics_informed_transformer import PhysicsInformedTransformer

def quick_test():
    """快速测试模型训练流程"""
    print("=== 快速测试模型训练流程 ===")
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        # 创建数据加载器（使用小批次进行快速测试）
        print("1. 加载数据...")
        train_loader, val_loader, test_loader, (input_scaler, output_scaler) = create_data_loaders(
            'dataset.csv', batch_size=64
        )
        print(f"   训练集大小: {len(train_loader.dataset)}")
        print(f"   验证集大小: {len(val_loader.dataset)}")
        print(f"   测试集大小: {len(test_loader.dataset)}")
        
        # 创建小型模型进行快速测试
        print("\n2. 创建模型...")
        model = PhysicsInformedTransformer(
            input_dim=8,
            output_dim=6,
            d_model=128,      # 减小模型大小
            n_heads=4,        # 减少注意力头
            n_layers=3,       # 减少层数
            d_ff=256,         # 减小前馈网络
            dropout=0.1
        )
        print(f"   模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        # 创建训练器
        print("\n3. 创建训练器...")
        trainer = ModelTrainer(model)
        
        # 快速训练（只训练几个epoch）
        print("\n4. 开始快速训练...")
        train_losses, val_losses = trainer.train(
            train_loader, val_loader,
            num_epochs=3,        # 只训练3个epoch
            learning_rate=1e-3,  # 稍大的学习率
            weight_decay=1e-5,
            patience=10
        )
        
        # 简单评估
        print("\n5. 评估模型...")
        metrics, predictions, targets = trainer.evaluate(test_loader, output_scaler)
        
        print("\n=== 快速测试结果 ===")
        for feature_name, metric in metrics.items():
            print(f"{feature_name}: R² = {metric['R2']:.4f}, RMSE = {metric['RMSE']:.6f}")
        
        print("\n✅ 模型训练流程测试成功！")
        print("现在可以运行完整训练: python training_pipeline.py")
        
    except Exception as e:
        print(f"❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_test()