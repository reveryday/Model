import torch
import logging
import numpy as np
from utils import inverse_transform_predictions
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.StreamHandler()])

def evaluate(model, test_loader, criterion, device, y_scaler):
    model.eval()
    test_loss = 0.0
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device) #[batch_size, 21] 
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            
            all_outputs.append(outputs.cpu())
            all_targets.append(targets.cpu())
    
    test_loss /= len(test_loader)
    all_outputs = torch.cat(all_outputs, dim=0) # [batch_size, data_length×test_size]
    all_targets = torch.cat(all_targets, dim=0)

    print(f"Log Targets max: {all_targets.max().item()} min: {all_targets.min().item()} mean: {all_targets.mean().item()}")
    print(f"Log Outputs max: {all_outputs.max().item()} min: {all_outputs.min().item()} mean: {all_outputs.mean().item()}")
#    print(f"Targets max: {all_targets.max().item()}")
#    print(f"Targets min: {all_targets.min().item()}")
#    print(f"Targets mean: {all_targets.mean().item()}")
#
#    print(f"outputs max: {all_outputs.max().item()}")
#    print(f"outputs min: {all_outputs.min().item()}")
#    print(f"outputs mean: {all_outputs.mean().item()}")


    
#    # 将预测结果和目标转换回原始尺度
#    if y_scaler is not None:
#        all_outputs_original = inverse_transform_predictions(all_outputs, y_scaler)  #[3600,3]
#        all_targets_original = inverse_transform_predictions(all_targets, y_scaler)  #[3600,3]
#
#        print(f"original Targets max: {all_targets_original.max().item()}")
#        print(f"original Targets min: {all_targets_original.min().item()}")
#        print(f"original Targets mean: {all_targets_original.mean().item()}")
#
#        print(f"original outputs max: {all_outputs_original.max().item()}")
#        print(f"original outputs min: {all_outputs_original.min().item()}")
#        print(f"original outputs mean: {all_outputs_original.mean().item()}")
#
#        
#        # 计算原始尺度下的MSE、MAE、MAPE
#        mse_original = np.mean((all_outputs_original - all_targets_original) ** 2)
#        mae_original = np.mean(np.abs(all_outputs_original - all_targets_original))
#        mape_original = np.mean(np.abs((all_targets_original - all_outputs_original) / all_targets_original)) * 100

    all_targets_original = np.expm1(all_targets.numpy()) 
    all_outputs_original = np.expm1(all_outputs.numpy())
    
    logging.info(f"original targets shape: {all_targets_original.shape}")
    logging.info(f"original Targets max: {all_targets_original.max().item()} min: {all_targets_original.min().item()} mean: {all_targets_original.mean().item()}")
    logging.info(f"original outputs shape: {all_outputs_original.shape}")
    logging.info(f"original outputs max: {all_outputs_original.max().item()} min: {all_outputs_original.min().item()} mean: {all_outputs_original.mean().item()}")
    
    mse = np.mean((all_outputs_original - all_targets_original) ** 2)
    mae = np.mean(np.abs(all_outputs_original - all_targets_original))
    mape = np.mean(np.abs((all_targets_original - all_outputs_original) / all_targets_original)) * 100
    r2 = r2_score(all_targets_original, all_outputs_original)

  
    logging.info(f'Test Loss: {test_loss:.6f}')
    logging.info(f'Test MAPE: {mape:.2f}%')
    logging.info(f'Test R2: {r2:.6f}')
    logging.info(f'Test MSE: {mse:.6f}')
    logging.info(f'Test MAE: {mae:.6f}')
    
    return test_loss, mse, mae, mape, r2

def draw_result(train_losses, val_losses, lr_history):

    # 绘制训练和验证损失
    plt.figure(figsize=(12, 6)) # 创建第一个figure对象
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.tight_layout()
    plt.savefig('train_val_loss.png') # 保存训练和验证损失图
    logging.info("Training and validation loss plot saved to train_val_loss.png")
    plt.close() # 关闭第一个figure对象，释放内存
    
    # 绘制学习率变化
    plt.figure(figsize=(12, 6)) # 创建第二个figure对象
    plt.plot(lr_history)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.tight_layout()
    plt.savefig('learning_rate.png') # 保存学习率变化图
    logging.info("Learning rate plot saved to learning_rate.png")
    plt.close() # 关闭第二个figure对象，释放内存