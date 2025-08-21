from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

RAW_DATA_DIR = Path(__file__).parent.parent / 'data' /'raw'
PROCESS_DATA_DIR = Path(__file__).parent.parent / 'data' /'processed'
LOGS_DIR = ROOT_DIR / 'logs'
MODELS_DIR = ROOT_DIR / 'models'

SEQ_LEN = 5
BATCH_SIZE = 128
EMBEDDING_DIM = 128
HIDDEN_SIZE=256

LEARNING_RATE = 0.001
EPOCHS = 10