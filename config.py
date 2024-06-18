# Training Hyperparameters
INPUT_SIZE = 784
NUM_CLASSES = 10
LEARNING_RATE = 0.001
NUM_EPOCHS = 3

# Dataset
DATA_DIR = "dataset/"
NUM_WORKERS = 4
BATCH_SIZE = 64

# Compute related
ACCELERATOR = "gpu"
DEVICES = [0]
PRECISION = 32
