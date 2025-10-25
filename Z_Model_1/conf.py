"""
parameters for the project
"""
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# load data
DATA_PATH = "./data.csv"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Y_SCALER = StandardScaler()
VAL_SIZE = 0.15 
TEST_SIZE = 0.15
RANDOM_STATE = 42
BATCH_SIZE=128

# model parameters
INPUT_DIM = 7
OUTPUT_DIM = 1
D_MODEL = 128
N_HEAD = 32
NUM_LAYERS = 10
D_FF = 512      
DROPOUT = 0.1   
MAX_NORM = 1  # 梯度裁剪的最大范数
ACTIVATION = "gelu"

# training parameters
num_epochs = 3
learning_rate = 1e-5
weight_decay = 5e-5 