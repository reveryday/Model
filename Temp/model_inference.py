import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from physics_informed_transformer import PhysicsInformedTransformer

class BUFPredictor:
    """伽马射线屏蔽累积因子预测器"""
    
    def __init__(self, model_path='physics_informed_transformer_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.input_scaler = None
        self.output_scaler = None
        self.feature_names = ['Inf_Flu_BUF', 'Fin_Flu_BUF', 'Inf_Exp_BUF', 
                             'Fin_Exp_BUF', 'Inf_Eff_BUF', 'Fin_Eff_BUF']
        self.input_names = ['Energy', 'Shell', 'MFP', 'MAC_Total', 'MAC_Incoherent', 
                           'MAC_Coherent', 'MAC_Photoelectric', 'MAC_Pair_production']
        
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """加载训练好的模型"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 创建模型
            self.model = PhysicsInformedTransformer(
                input_dim=8,
                output_dim=6,
                d_model=256,
                n_heads=8,
                n_layers=6,
                d_ff=1024,
                dropout=0.1
            )
            
            # 加载模型权重
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # 加载预处理器
            self.input_scaler = checkpoint['input_scaler']
            self.output_scaler = checkpoint['output_scaler']
            
            print(f"模型已成功加载到 {self.device}")
            
        except FileNotFoundError:
            print(f"模型文件 {model_path} 未找到，请先训练模型")
        except Exception as e:
            print(f"加载模型时出错: {e}")
    
    def predict_single(self, energy, shell, mfp, mac_total, mac_incoherent, 
                      mac_coherent, mac_photoelectric, mac_pair_production):
        """预测单个样本的累积因子"""
        if self.model is None:
            raise ValueError("模型未加载，请先加载模型")
        
        # 准备输入数据
        input_data = np.array([[energy, shell, mfp, mac_total, mac_incoherent, 
                               mac_coherent, mac_photoelectric, mac_pair_production]])
        
        # 标准化
        input_scaled = self.input_scaler.transform(input_data)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32).to(self.device)
        
        # 预测
        with torch.no_grad():
            output, attention_weights = self.model(input_tensor)
            output_scaled = output.cpu().numpy()
        
        # 反标准化
        predictions = self.output_scaler.inverse_transform(output_scaled)
        
        # 返回结果字典
        result = {}
        for i, name in enumerate(self.feature_names):
            result[name] = float(predictions[0, i])
        
        return result, attention_weights
    
    def predict_batch(self, input_data):
        """批量预测"""
        if self.model is None:
            raise ValueError("模型未加载，请先加载模型")
        
        # 如果输入是DataFrame，转换为numpy数组
        if isinstance(input_data, pd.DataFrame):
            input_array = input_data.values
        else:
            input_array = np.array(input_data)
        
        # 标准化
        input_scaled = self.input_scaler.transform(input_array)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32).to(self.device)
        
        # 预测
        with torch.no_grad():
            output, attention_weights = self.model(input_tensor)
            output_scaled = output.cpu().numpy()
        
        # 反标准化
        predictions = self.output_scaler.inverse_transform(output_scaled)
        
        # 返回DataFrame
        result_df = pd.DataFrame(predictions, columns=self.feature_names)
        
        return result_df, attention_weights
    
    def analyze_energy_dependence(self, shell=0, mfp_values=[1.0, 5.0, 10.0], 
                                 energy_range=(0.01, 10.0), num_points=100):
        """分析累积因子随能量的变化"""
        energies = np.linspace(energy_range[0], energy_range[1], num_points)
        
        # 使用数据集中的典型MAC值（这里使用平均值作为示例）
        typical_mac = {
            'MAC_Total': 0.5,
            'MAC_Incoherent': 0.1,
            'MAC_Coherent': 0.05,
            'MAC_Photoelectric': 0.3,
            'MAC_Pair_production': 0.05
        }
        
        results = {}
        
        for mfp in mfp_values:
            predictions = []
            
            for energy in energies:
                pred, _ = self.predict_single(
                    energy=energy,
                    shell=shell,
                    mfp=mfp,
                    **typical_mac
                )
                predictions.append(pred)
            
            results[f'MFP_{mfp}'] = predictions
        
        # 绘制结果
        self.plot_energy_dependence(energies, results)
        
        return energies, results
    
    def analyze_thickness_dependence(self, energy=1.0, shell=0, 
                                   thickness_range=(0.0, 20.0), num_points=50):
        """分析累积因子随厚度的变化"""
        thicknesses = np.linspace(thickness_range[0], thickness_range[1], num_points)
        
        # 使用典型MAC值
        typical_mac = {
            'MAC_Total': 0.5,
            'MAC_Incoherent': 0.1,
            'MAC_Coherent': 0.05,
            'MAC_Photoelectric': 0.3,
            'MAC_Pair_production': 0.05
        }
        
        predictions = []
        
        for thickness in thicknesses:
            pred, _ = self.predict_single(
                energy=energy,
                shell=shell,
                mfp=thickness,
                **typical_mac
            )
            predictions.append(pred)
        
        # 绘制结果
        self.plot_thickness_dependence(thicknesses, predictions)
        
        return thicknesses, predictions
    
    def plot_energy_dependence(self, energies, results):
        """绘制能量依赖性图"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, feature in enumerate(self.feature_names):
            for j, (mfp_label, predictions) in enumerate(results.items()):
                values = [pred[feature] for pred in predictions]
                axes[i].plot(energies, values, label=mfp_label, 
                           color=colors[j % len(colors)], linewidth=2)
            
            axes[i].set_xlabel('能量 (MeV)')
            axes[i].set_ylabel(feature)
            axes[i].set_title(f'{feature} vs 能量')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xscale('log')
            axes[i].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig('energy_dependence_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_thickness_dependence(self, thicknesses, predictions):
        """绘制厚度依赖性图"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, feature in enumerate(self.feature_names):
            values = [pred[feature] for pred in predictions]
            axes[i].plot(thicknesses, values, 'b-', linewidth=2, label=feature)
            
            axes[i].set_xlabel('厚度 (MFP)')
            axes[i].set_ylabel(feature)
            axes[i].set_title(f'{feature} vs 厚度')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig('thickness_dependence_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_attention_weights(self, energy, shell, mfp, mac_total, mac_incoherent, 
                                  mac_coherent, mac_photoelectric, mac_pair_production):
        """可视化注意力权重"""
        _, attention_weights = self.predict_single(
            energy, shell, mfp, mac_total, mac_incoherent, 
            mac_coherent, mac_photoelectric, mac_pair_production
        )
        
        # 取最后一层的注意力权重
        last_layer_attention = attention_weights[-1][0].cpu().numpy()  # [n_heads, seq_len, seq_len]
        
        # 平均所有注意力头
        avg_attention = np.mean(last_layer_attention, axis=0)
        
        # 绘制注意力热图
        plt.figure(figsize=(10, 8))
        sns.heatmap(avg_attention, 
                   xticklabels=self.input_names,
                   yticklabels=self.input_names,
                   annot=True, 
                   cmap='Blues',
                   fmt='.3f')
        plt.title('注意力权重热图')
        plt.xlabel('输入特征')
        plt.ylabel('输入特征')
        plt.tight_layout()
        plt.savefig('attention_weights.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def compare_with_ground_truth(self, test_csv_file, num_samples=1000):
        """与真实值比较"""
        # 读取测试数据
        test_data = pd.read_csv(test_csv_file)
        
        # 随机选择样本
        if len(test_data) > num_samples:
            test_data = test_data.sample(n=num_samples, random_state=42)
        
        # 预测
        input_data = test_data.iloc[:, :8]
        true_values = test_data.iloc[:, 8:].values
        
        predictions, _ = self.predict_batch(input_data)
        pred_values = predictions.values
        
        # 计算误差
        errors = np.abs(pred_values - true_values) / true_values * 100  # 相对误差百分比
        
        # 绘制误差分布
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, feature in enumerate(self.feature_names):
            axes[i].hist(errors[:, i], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            axes[i].set_xlabel('相对误差 (%)')
            axes[i].set_ylabel('频次')
            axes[i].set_title(f'{feature} 相对误差分布')
            axes[i].axvline(np.mean(errors[:, i]), color='red', linestyle='--', 
                           label=f'平均误差: {np.mean(errors[:, i]):.2f}%')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('error_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 打印统计信息
        print("=== 预测误差统计 ===")
        for i, feature in enumerate(self.feature_names):
            print(f"{feature}:")
            print(f"  平均相对误差: {np.mean(errors[:, i]):.3f}%")
            print(f"  中位数相对误差: {np.median(errors[:, i]):.3f}%")
            print(f"  95%分位数误差: {np.percentile(errors[:, i], 95):.3f}%")
            print()

def interactive_prediction():
    """交互式预测界面"""
    predictor = BUFPredictor()
    
    if predictor.model is None:
        print("模型加载失败，请检查模型文件")
        return
    
    print("=== 伽马射线屏蔽累积因子预测器 ===")
    print("请输入以下参数：")
    
    try:
        energy = float(input("能量 (MeV): "))
        shell = int(input("壳层数: "))
        mfp = float(input("厚度 (MFP): "))
        mac_total = float(input("总衰减系数: "))
        mac_incoherent = float(input("非相干散射衰减系数: "))
        mac_coherent = float(input("相干散射衰减系数: "))
        mac_photoelectric = float(input("光电效应衰减系数: "))
        mac_pair_production = float(input("电子对产生衰减系数: "))
        
        # 预测
        result, _ = predictor.predict_single(
            energy, shell, mfp, mac_total, mac_incoherent,
            mac_coherent, mac_photoelectric, mac_pair_production
        )
        
        print("\n=== 预测结果 ===")
        for name, value in result.items():
            print(f"{name}: {value:.6f}")
        
        # 可视化注意力权重
        print("\n是否显示注意力权重? (y/n): ", end="")
        if input().lower() == 'y':
            predictor.visualize_attention_weights(
                energy, shell, mfp, mac_total, mac_incoherent,
                mac_coherent, mac_photoelectric, mac_pair_production
            )
        
    except ValueError:
        print("输入格式错误，请输入数字")
    except Exception as e:
        print(f"预测过程中出错: {e}")

if __name__ == "__main__":
    # 创建预测器
    predictor = BUFPredictor()
    
    if predictor.model is not None:
        print("模型加载成功！")
        
        # 示例：分析能量依赖性
        print("\n分析累积因子随能量的变化...")
        predictor.analyze_energy_dependence()
        
        # 示例：分析厚度依赖性
        print("\n分析累积因子随厚度的变化...")
        predictor.analyze_thickness_dependence()
        
        # 示例：与真实值比较
        print("\n与真实值比较...")
        predictor.compare_with_ground_truth('dataset.csv', num_samples=1000)
        
        # 交互式预测
        print("\n启动交互式预测...")
        interactive_prediction()
    else:
        print("请先运行 training_pipeline.py 训练模型")