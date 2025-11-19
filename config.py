from pathlib import Path
import torch

# Path Configuration
DATA_PATH = Path(r"data\IWSLT-15-en-vi")

# TOKENIZER_NAME = ""
# TOKENIZER_NAME = "iwslt_en-vi_tokenizer_16k.json"
TOKENIZER_NAME = "iwslt_en-vi_tokenizer_32k.json"
TOKENIZER_PATH = Path(r"artifacts\tokenizers") / TOKENIZER_NAME


# MODEL_NAME = ""
# MODEL_NAME = "transformer_en_vi_iwslt_1.pt"
MODEL_NAME = "transformer_en_vi_iwslt_1.safetensors"
MODEL_SAVE_PATH = Path(r"artifacts\models") / MODEL_NAME

CHECKPOINT_PATH = Path(r"artifacts\checkpoints") / MODEL_NAME

CACHE_DIR = ""

# Hardware & Data Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VOCAB_SIZE: int = 32_000

SPECIAL_TOKENS: list[str] = ["[PAD]", "[UNK]", "[SOS]", "[EOS]"]

NUM_WORKERS: int = 8

# Tokenizer Constants
PAD_TOKEN_ID: int = 0
UNK_TOKEN_ID: int = 1
SOS_TOKEN_ID: int = 2
EOS_TOKEN_ID: int = 3

# Model Hyperparameters
D_MODEL: int = 256  # (Dimension of model)
N_LAYERS: int = 6  # (N=6 in paper)
N_HEADS: int = 8  # (h=8 in paper)
D_FF: int = 1024  # (d_ff = 4 * d_model = 1024)
DROPOUT: float = 0.1  # (Dropout = 0.1 in paper)
MAX_SEQ_LEN: int = 100  # (Max length for Positional Encoding)

NUM_SAMPLES_TO_USE: int = 1000
# NUM_SAMPLE_TO_USE: int = 1_000_000

# Training Configuration
LEARNING_RATE: float = 1e-4
BATCH_SIZE: int = 32
EPOCHS: int = 5

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
