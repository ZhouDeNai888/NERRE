
# Training Configuration for Graph RAG NER+RE

MODEL_NAME = "xlm-roberta-base"

# Loss Hyperparameters (Focal Loss)
ALPHA = 0.8
GAMMA = 2.0

# Asymmetric Focal Loss Hyperparameters
POS_GAMMA = 1.0  # สำหรับ Entity
NEG_GAMMA = 4.0  # สำหรับ คลาส O

# Optimizer Hyperparameters
LR = 5e-5
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

# Training Loop Hyperparameters
NUM_EPOCHS = 7  # ✅ 15 epochs with label_smoothing
BATCH_SIZE = 128  # ✅ Increased batch size for better stability

# Output
OUTPUT_DIR = "saved_model_v16"  # ✅ Version 13 - Multilingual with correct positions

# Dataset - Multilingual dataset v4 (10000 samples)
TRAIN_FILE = "dataset/multilingual_data_v7_50000.json"


VAL_FILE = "dataset/data_v2.json" # Using original val set for evaluation