
# Training Configuration for Graph RAG NER+RE

MODEL_NAME = "xlm-roberta-base"

# Loss Hyperparameters (Focal Loss)
ALPHA = 0.25
ALPHA_rel = 0.95
GAMMA = 2.0

# Optimizer Hyperparameters
LR = 5e-5
WEIGHT_DECAY = 0.05
MAX_GRAD_NORM = 1.0

# Training Loop Hyperparameters
NUM_EPOCHS = 15  # ✅ 15 epochs with label_smoothing
BATCH_SIZE = 4  # ✅ Increased batch size for better stability

# Output
OUTPUT_DIR = "saved_model_v25" 

# Dataset - Multilingual dataset v4 (10000 samples)
TRAIN_FILE = ["/data/tcustpg18/NERRE/NERRE/dataset/multilingual_data_v8_train.json",]

VAL_FILE = ["/data/tcustpg18/NERRE/NERRE/dataset/multilingual_data_v8_val.json",]