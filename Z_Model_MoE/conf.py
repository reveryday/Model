"""
Configuration parameters for Shell-Conditional MoE Model
"""
import torch
from sklearn.preprocessing import MinMaxScaler

# Data loading
DATA_PATH = "./BUF_DATA_with_MAC_no_material.csv"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Y_SCALER = MinMaxScaler()
VAL_SIZE = 0.15 
TEST_SIZE = 0.15
RANDOM_STATE = 42
BATCH_SIZE = 128

# Model parameters - MoE specific
INPUT_DIM = 8  # Extended to include all MAC components
OUTPUT_DIM = 6
NUM_SHELLS = 102  # Maximum shell index + 1 (based on data analysis)
EMBED_DIM = 32   # Shell embedding dimension
NUM_EXPERTS = 6  # Number of expert networks
EXPERT_HIDDEN_DIM = 256  # Expert network hidden dimension
TRUNK_WIDTH = 256  # Shared trunk network width
TRUNK_DEPTH = 3    # Shared trunk network depth
DROPOUT = 0.1
MAX_NORM = 1.0

# Backbone (Transformer) parameters
BACKBONE_D_MODEL = 256
BACKBONE_NHEAD = 8
BACKBONE_LAYERS = 4
BACKBONE_DIM_FEEDFORWARD = 512

# Physics constraint parameters
PHYSICS_WEIGHT = 0.2  # Weight for physics constraint loss
PHYSICS_LOSS_WEIGHT = 0.2  # Alternative name for compatibility
PHYSICS_CUMULATIVE_WEIGHT = 1.0
PHYSICS_GEOMETRY_WEIGHT = 1.0
PHYSICS_MONOTONICITY_WEIGHT = 0.1
MIN_BUF_VALUE = 1.0   # Minimum buildup factor value
MONOTONICITY_WEIGHT = 0.1  # Weight for thickness monotonicity constraint

# Training parameters
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
PATIENCE = 30  # Early stopping patience
EARLY_STOPPING_PATIENCE = 30  # Alternative name for compatibility
T_0 = 20       # CosineAnnealingWarmRestarts initial period
T_MULT = 2     # Period multiplication factor
ETA_MIN = 1e-6 # Minimum learning rate
NOISE_LEVEL = 0.01  # Noise level for data augmentation
GRADIENT_CLIP_VALUE = 1.0  # Gradient clipping value

# Feature columns (8 dimensions)
FEATURE_COLUMNS = [
    'Energy', 'Shell', 'MFP', 'MAC_Total', 
    'MAC_Incoherent', 'MAC_Coherent', 
    'MAC_Photoelectric', 'MAC_Pair_production'
]

# Target columns (6 dimensions)
TARGET_COLUMNS = [
    'Inf_Flu_BUF', 'Fin_Flu_BUF', 'Inf_Exp_BUF', 
    'Fin_Exp_BUF', 'Inf_Eff_BUF', 'Fin_Eff_BUF'
]

# Evaluation settings
EVAL_BY_SHELL = True  # Whether to evaluate performance by shell
SHELL_GROUPS = [
    (0, 10), (11, 30), (31, 50), (51, 70), (71, 101)
]  # Shell groups for analysis