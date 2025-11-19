from pathlib import Path
from datasets import Dataset
from tokenizers import (
    Tokenizer,
    models,
    normalizers,
    pre_tokenizers,
    decoders,
    trainers,
)
from tqdm.auto import tqdm
from utils import get_raw_data


DATA_PATH = Path(r"..\data\IWSLT-15-en-vi")
# TOKENIZER_NAME = "iwslt_en-vi_tokenizer_16k.json"
TOKENIZER_NAME = "iwslt_en-vi_tokenizer_32k.json"
TOKENIZER_SAVE_PATH = Path(r"..\artifacts\tokenizers") / TOKENIZER_NAME

# VOCAB_SIZE: int = 16_000
VOCAB_SIZE: int = 32_000
SPECIAL_TOKENS: list[str] = ["[PAD]", "[UNK]", "[SOS]", "[EOS]"]

BATCH_SIZE_FOR_TOKENIZER: int = 100000
NUM_WORKERS: int = 8


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

    train_dataset = get_raw_data(DATA_PATH, for_tokenizer=True)
    if not isinstance(train_dataset, Dataset):
        train_dataset = Dataset.from_list(train_dataset)
    print(f"Starting tokenizer training on {len(train_dataset)} pairs...")

    # 1. Define the iterator AND batch size
    text_iterator = get_training_corpus(
        train_dataset,
        batch_size=BATCH_SIZE_FOR_TOKENIZER,
    )

    # 2. Calculate total steps for the progress bar
    total_steps = (len(train_dataset) // BATCH_SIZE_FOR_TOKENIZER) * 2
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
