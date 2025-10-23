import torch
import torch.nn as nn
import torch.optim as optim
import logging
import time
import os
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau

from model import MyModel, MLPModel
from utils import load_data, AddNoise, HuberLoss
from evaluate import evaluate, draw_result
from conf import *
import warnings

warnings.filterwarnings("ignore", message="Flash attention was not supported")
warnings.filterwarnings("ignore", message="memory efficient attention")

logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.StreamHandler()])

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device):
    
    train_losses = []
    val_losses = []
    lr_history = []
    
    best_val_loss = float('inf')
    best_model_state = None
    patience = 500
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)            
            # 添加适当的噪声可以提高模型的鲁棒性
            inputs = AddNoise(inputs, noise_level=0.01)  # 增加噪声水平以提高鲁棒性            
            optimizer.zero_grad()            
            outputs = model(inputs)
            loss = criterion(outputs, targets)  # L1Loss默认把损失计算为“所有元素差的绝对值取平均          
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_NORM)
            optimizer.step()
            train_loss += loss.item()
        
        # 计算训练损失
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        model.eval() #将模型设置为评估模式：关闭 dropout等训练特定的行为
        val_loss = 0.0
        # 计算验证损失：评估模型在未见过的数据上的泛化能力、监控训练过程
        with torch.no_grad():
            if torch.isnan(outputs).any():
                print("NaN in model output!")
                break
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy() #保存参数
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 早停
        if patience_counter >= patience:
            logging.info(f'Early stopping at epoch {epoch+1}')
            break
        
        # 更新学习率调度器
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
        else:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
    
        lr_history.append(current_lr)
        
        logging.info(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.6f}')
    
    # 加载最佳模型参数
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logging.info(f'Loaded best model with validation loss: {best_val_loss:.6f}')
    
    return model, train_losses, val_losses, lr_history

def main():    
    # 记录程序开始时间
    total_start_time = time.time()    
    logging.info(f'device: {device}')
    
    # 加载数据
    data_dict = load_data()   
    train_loader = data_dict['train_loader']
    val_loader = data_dict['val_loader']
    test_loader = data_dict['test_loader']
    y_scaler = data_dict['y_scaler']
    
    model = MyModel().to(device)
    #model = MLPModel().to(device)

    # 加载已保存的模型权重（如果存在best model.pth文件）
    model_path = './outputs/checkpoints/best_model.pth'
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint)
            logging.info(f'Successfully loaded pretrained model from: {model_path}')
        except Exception as e:
            logging.error(f'Error loading pretrained model: {e}')
            logging.info('Starting training from scratch...')
    else:
        logging.info('No pretrained model found, starting training from scratch...')

    #criterion = HuberLoss(delta=0.5) # 使用Huber Loss替代L1 Loss
    criterion = nn.L1Loss()
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        amsgrad=True
    )

    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=20,  # 初始周期
        T_mult=2,  # 每次重启后周期长度乘以的因子
        eta_min=1e-6  # 最小学习率
    )
    
    model, train_losses, val_losses, lr_history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device
    )
    
    test_loss, MSE, MAE, MAPE, R2 = evaluate(
        model=model, 
        test_loader=test_loader, 
        criterion=criterion, 
        device=device, 
        y_scaler=y_scaler)
    
    # 保存模型
    os.makedirs('./outputs', exist_ok=True)
    os.makedirs('./outputs/checkpoints', exist_ok=True)
    torch.save(model.state_dict(), './outputs/checkpoints/best_model.pth')
    logging.info("Model saved to outputs/checkpoints/best_model.pth")
    
    # 计算程序总运行时间
    total_end_time = time.time()
    total_run_time = total_end_time - total_start_time
    hours, remainder = divmod(total_run_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logging.info(f'Total program execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s')
    
    # 绘制回归曲线、学习率变化图
    draw_result(train_losses, val_losses, lr_history)

if __name__ == '__main__':
    main()