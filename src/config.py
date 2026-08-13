from pathlib import Path
import torch

# dirs
PROJECT_DIR = Path(__file__).resolve().parent.parent

CHECKPOINTS_DIR = PROJECT_DIR / "checkpoints"
DATA_DIR = PROJECT_DIR / "data"
REPORTS_DIR = PROJECT_DIR / "reports"
SRC_DIR = PROJECT_DIR / "src"

# data
TRAIN_DATA = DATA_DIR / "train"
VAL_DATA = DATA_DIR / "val"
TEST_DATA = DATA_DIR / "test" / "images"
WORDS_FILE = DATA_DIR / "words.txt"

# configs
IMAGE_SIZE = 64
BATCH_SIZE = 256
NUM_CLASSES = 200
EPOCHS = 100
LR = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = torch.cuda.is_available()
NUM_WORKERS = 0