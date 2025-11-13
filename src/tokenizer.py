from pathlib import Path
import datasets
from datasets import Dataset, load_dataset
from tokenizers import (
    Tokenizer,
    models,
    normalizers,
    pre_tokenizers,
    decoders,
    trainers,
)
from tqdm.auto import tqdm
from utils import is_valid_pair, filter_empty


DATA_PATH = Path(r"..\data\IWSLT-15-en-vi")
TOKENIZER_NAME = "iwslt_en-vi_tokenizer_16k.json"
TOKENIZER_SAVE_PATH = Path(r"..\artifacts\tokenizers") / TOKENIZER_NAME

VOCAB_SIZE: int = 32_000
SPECIAL_TOKENS: list[str] = ["[PAD]", "[UNK]", "[SOS]", "[EOS]"]

BATCH_SIZE_FOR_TOKENIZER: int = 10000
NUM_WORKERS: int = 8


def get_raw_data() -> Dataset:
    print("Loading datasets from local Parquet files...")
    # 1. Load all 3 splits from the local Parquet files
    all_splits: datasets.DatasetDict = load_dataset(path=str(DATA_PATH))

    print(all_splits)

    # 2. Assign to our standard variable names
    # train_data_raw = all_splits["train"]
    # val_data_raw = all_splits["validation"]
    # test_data_raw = all_splits["test"]

    # print(f"\nDatasets loaded successfully:")
    # print(f"  Train samples: {len(train_data_raw)}")
    # print(f"  Val samples: {len(val_data_raw)}")
    # print(f"  Test samples: {len(test_data_raw)}")
    # print("\nExample (from train split):")
    # print(train_data_raw[0])
    # print(val_data_raw[0])
    # print(test_data_raw[0])

    print("--- Filtering Datasets (Removing empty sentences) ---")
    train_raw_data: Dataset = filter_empty(all_splits["train"], NUM_WORKERS)

    return train_raw_data


def get_training_corpus(dataset: Dataset, batch_size: int = 1000):
    """
    A generator function to yield batches of text.

    This implementation uses dataset.iter(batch_size=...), which is the
    highly optimized, zero-copy Arrow iterator.

    We then use list comprehensions to extract the 'en' and 'vi' strings
    from the nested list of dictionaries returned by the iterator.
    """

    # We iterate over the dataset in batches
    # batch will be: {'translation': [list of 1000 dicts]}
    for batch in dataset.iter(batch_size=batch_size):

        # We must iterate through the list 'batch['translation']'
        # to extract the individual strings.

        # This list comprehension is fast and Pythonic.
        en_strings: list[str] = [item["en"] for item in batch["translation"]]
        vi_strings: list[str] = [item["vi"] for item in batch["translation"]]

        # Yield the batch of strings (which the trainer expects)
        yield en_strings
        yield vi_strings


def instantiate_tokenizer() -> Tokenizer:
    # 1. Initialize an empty Tokenizer with a BPE model
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

    # 2. (Best Practice) Set up the normalizer and pre-tokenizer
    # Normalizer: Cleans the text (e.g., Unicode, lowercase)
    tokenizer.normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),  # Unicode normalization
            normalizers.Lowercase(),  # Convert to lowercase
        ]
    )

    # Pre-tokenizer: Splits text into "words" (e.g., by space, punctuation)
    # BPE will then learn to merge sub-words from these.
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # Decoder: Reconstructs the string from tokens
    tokenizer.decoder = decoders.BPEDecoder()

    print("Tokenizer (empty) initialized.")
    return tokenizer


def train_tokenizer():
    # Initialize the BpeTrainer
    trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=SPECIAL_TOKENS)

    print("Tokenizer Trainer initialized.")

    train_raw_data = get_raw_data()
    print(f"Starting tokenizer training on {len(train_raw_data)} pairs...")

    # 1. Define the iterator AND batch size
    BATCH_SIZE_FOR_TOKENIZER = 10000
    text_iterator = get_training_corpus(
        train_raw_data,
        batch_size=BATCH_SIZE_FOR_TOKENIZER,
    )

    # 2. (Best Practice) Calculate total steps for the progress bar
    total_steps = (len(train_raw_data) // BATCH_SIZE_FOR_TOKENIZER) * 2
    if total_steps == 0:
        total_steps = 1  # (Avoid division by zero if dataset is tiny)

    tokenizer: Tokenizer = instantiate_tokenizer()
    # 3. Train with tqdm progress bar
    try:
        tokenizer.train_from_iterator(
            tqdm(
                text_iterator,
                total=total_steps,
                desc="Training Tokenizer (IWSLT-Local)",
            ),
            trainer=trainer,
            length=total_steps,
        )
    except KeyboardInterrupt:
        print("\nTokenizer training interrupted by user.")

    print("Tokenizer training complete.")

    tokenizer.save(str(TOKENIZER_SAVE_PATH))

    print(f"Tokenizer saved to: {TOKENIZER_SAVE_PATH}")
    print(f"Total vocabulary size: {tokenizer.get_vocab_size()}")


if __name__ == "__main__":
    # dataset = get_raw_data()
    # print(type(dataset))

    # tokenizer: Tokenizer = instantiate_tokenizer()
    # tokenizer.save(str(TOKENIZER_SAVE_PATH))

    train_tokenizer()
