
# Training Configuration for Graph RAG NER+RE

MODEL_NAME = "xlm-roberta-base"

# Loss Hyperparameters (Focal Loss)
ALPHA = 0.25
GAMMA = 2.0

# Optimizer Hyperparameters
LR = 2e-5
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

# Training Loop Hyperparameters
NUM_EPOCHS = 50  # ✅ 50 epochs with label_smoothing
BATCH_SIZE = 8    
STEPS_PER_EPOCH = 100

# Output
OUTPUT_DIR = "saved_model_v10"  # ✅ Version 10 - Multilingual with correct positions

# Dataset - Multilingual dataset v2 (163 samples: 131 EN + 32 multilingual)
TRAIN_FILE = "dataset/combined_multilingual_v2.json"
