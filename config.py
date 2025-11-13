from pathlib import Path
import torch

# Path Configuration
DATA_PATH = Path(r"..\data\IWSLT-15-en-vi")

# TOKENIZER_NAME = ""
TOKENIZER_NAME = "iwslt_en-vi_tokenizer_16k.json"
TOKENIZER_PATH = Path(r"artifacts\tokenizers") / TOKENIZER_NAME

# MODEL_NAME = ""
MODEL_NAME = "transformer_en_vi_iwslt_1"
MODEL_SAVE_PATH = Path(r"artifacts\models") / MODEL_NAME

CACHE_DIR = ""


# Hardware & Data Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VOCAB_SIZE: int = 32_000

SPECIAL_TOKENS: list[str] = ["[PAD]", "[UNK]", "[SOS]", "[EOS]"]

NUM_WORKERS: int = 4

NUM_SAMPLE_TO_USE: int = 1_000_000

BATCH_SIZE: int = 32


# Tokenizer Constants
PAD_TOKEN_ID: int = 0
SOS_TOKEN_ID: int = 2
EOS_TOKEN_ID: int = 3


# Model Hyperparameters
D_MODEL: int = 256  # (Dimension of model)
N_LAYERS: int = 6  # (N=6 in paper)
N_HEADS: int = 8  # (h=8 in paper)
D_FF: int = 1024  # (d_ff = 4 * d_model = 1024)
DROPOUT: float = 0.1  # (Dropout = 0.1 in paper)
MAX_SEQ_LEN: int = 100  # (Max length for Positional Encoding)


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
