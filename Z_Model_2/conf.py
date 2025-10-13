"""
parameters for the project
"""
import torch
from sklearn.preprocessing import MinMaxScaler

# load data
# DATA_PATH = "./dataset_reduced.csv"
DATA_PATH = "./BUF_DATA_with_MAC_no_material.csv"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Y_SCALER = MinMaxScaler()
VAL_SIZE = 0.15 
TEST_SIZE = 0.15
RANDOM_STATE = 42
BATCH_SIZE=128

# model parameters
INPUT_DIM = 4
OUTPUT_DIM = 6
D_MODEL = 128
N_HEAD = 32
NUM_LAYERS = 10
D_FF = 512      
DROPOUT = 0.1   
MAX_NORM = 1.0
ACTIVATION = "gelu"

# training parameters
num_epochs = 5
learning_rate = 1e-5
weight_decay = 5e-5 