from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = ROOT_DIR/"data"/"raw"
PROCESS_DATA_DIR = ROOT_DIR/"data"/"process"
LOGS_DIR = ROOT_DIR/"logs"
MODELS_DIR = ROOT_DIR/"models"

# SEQ_LEN = 128
BATCH_SIZE = 64
EMBEDDING_DIM = 128
HIDDEN_SIZE = 256
LEARNING_RATE = 1e-3
EPOCHS = 50
MAX_SEQ_LENGTH = 128