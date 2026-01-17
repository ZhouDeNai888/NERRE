
# Training Configuration

MODEL_NAME = "xlm-roberta-base"

# Loss Hyperparameters
ALPHA = 0.25
GAMMA = 2.0

# Optimizer Hyperparameters
LR = 2e-5
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

# Training Loop Hyperparameters
NUM_EPOCHS = 5
BATCH_SIZE = 16
STEPS_PER_EPOCH = 100

# Output
OUTPUT_DIR = "saved_model_v1"

TRAIN_FILE = "dataset/data.json"

