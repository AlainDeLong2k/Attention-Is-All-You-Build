import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as ArrowDataset
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

import config
from src import utils


class TranslationDataset(Dataset):
    """
    A "lazy" Dataset.
    Uses the high-level PreTrainedTokenizerFast wrapper.
    """

    def __init__(
        self,
        dataset: ArrowDataset,
        tokenizer: PreTrainedTokenizerFast,
        max_len_src: int,
        max_len_tgt: int,
        src_lang: str = "en",
        tgt_lang: str = "vi",
    ):
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_len_src = max_len_src
        self.max_len_tgt = max_len_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, list[int]]:

        item = self.dataset[index]["translation"]
        src_text = item[self.src_lang]
        tgt_text = item[self.tgt_lang]

        # We set add_special_tokens=False for manual control.
        src_encoding = self.tokenizer(
            src_text,
            truncation=True,
            max_length=self.max_len_src,
            add_special_tokens=False,  # (Source has no SOS/EOS)
        )

        tgt_encoding = self.tokenizer(
            tgt_text,
            truncation=True,
            max_length=self.max_len_tgt - 2,  # (Reserve 2 spots for SOS/EOS)
            add_special_tokens=False,
        )

        # Manually add SOS/EOS to target
        src_ids = src_encoding["input_ids"]

        tgt_ids = (
            [config.SOS_TOKEN_ID] + tgt_encoding["input_ids"] + [config.EOS_TOKEN_ID]
        )

        return {"src_ids": src_ids, "tgt_ids": tgt_ids}


class DataCollator:
    """
    Implements a custom collate_fn.

    1. Takes a list of dicts (from __getitem__)
    2. Adds SOS/EOS (Wait, we did this in Dataset)
    3. Creates decoder inputs and labels (shifted)
    4. Dynamically pads all sequences *in the batch*
    5. Creates all 3 required masks
    6. Returns a single dict of tensors
    """

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict[str, list[int]]]) -> dict[str, Tensor]:

        # 1. Get raw ID lists from the batch
        src_ids_list = [item["src_ids"] for item in batch]
        tgt_ids_list = [item["tgt_ids"] for item in batch]  # (Already has SOS/EOS)

        # 2. Create shifted inputs/labels
        # Decoder input (T_tgt): [SOS, w1, w2, w3]
        dec_input_ids_list = [ids[:-1] for ids in tgt_ids_list]
        # Label (T_tgt): [w1, w2, w3, EOS]
        labels_list = [ids[1:] for ids in tgt_ids_list]

        # 3. Dynamic Padding
        # We use torch.nn.utils.rnn.pad_sequence
        # (Note: batch_first=True means (B, T))
        src_ids_padded = nn.utils.rnn.pad_sequence(
            [torch.tensor(ids) for ids in src_ids_list],
            batch_first=True,
            padding_value=self.pad_token_id,
        )

        dec_input_ids_padded = nn.utils.rnn.pad_sequence(
            [torch.tensor(ids) for ids in dec_input_ids_list],
            batch_first=True,
            padding_value=self.pad_token_id,
        )

        labels_padded = nn.utils.rnn.pad_sequence(
            [torch.tensor(ids) for ids in labels_list],
            batch_first=True,
            padding_value=self.pad_token_id,  # (Loss will ignore this ID)
        )

        # 4. Get the sequence length
        _, T_tgt = dec_input_ids_padded.shape

        # 5. Create Masks (on CPU)

        # (Mask 1) Source padding mask (for Encoder MHA & Cross-Attn)
        # Shape: (B, 1, 1, T_src)
        src_mask = utils.create_padding_mask(src_ids_padded, self.pad_token_id)

        # (Mask 2) Target padding mask (for Decoder MHA)
        # Shape: (B, 1, 1, T_tgt)
        tgt_padding_mask = utils.create_padding_mask(
            dec_input_ids_padded, self.pad_token_id
        )

        # (Mask 3) Target look-ahead mask (for Decoder MHA)
        # Shape: (1, 1, T_tgt, T_tgt)
        look_ahead_mask = utils.create_look_ahead_mask(T_tgt)

        # (Mask 4) Combined target mask
        # Shape: (B, 1, T_tgt, T_tgt)
        tgt_mask = tgt_padding_mask & look_ahead_mask

        return {
            "src_ids": src_ids_padded,  # (B, T_src)
            "tgt_input_ids": dec_input_ids_padded,  # (B, T_tgt)
            "labels": labels_padded,  # (B, T_tgt)
            "src_mask": src_mask,  # (B, 1, 1, T_src)
            "tgt_mask": tgt_mask,  # (B, 1, T_tgt, T_tgt)
        }


def get_translation_datasets(
    tokenizer: PreTrainedTokenizerFast,
) -> tuple[TranslationDataset, TranslationDataset, TranslationDataset]:
    """
    A Factory function to automate the data pipeline setup.

    It performs 3 steps:
    1. Loads and cleans raw data (using src.utils).
    2. Instantiates the TranslationDataset for Train, Val, and Test splits.
    3. Returns the 3 PyTorch datasets ready for the DataLoader.

    Args:
        tokenizer: The trained tokenizer.

    Returns:
        Tuple containing (train_ds, val_ds, test_ds)
    """

    # 1. Load raw cleaned data (returns Dict[str, Dataset])
    #    This keeps train.py clean from raw data handling logic.
    train_data, val_data, test_data = utils.get_raw_data(
        config.DATA_PATH, num_workers=config.NUM_WORKERS
    )
    train_data = train_data.select(range(config.NUM_SAMPLES_TO_USE))

    print(f"Building PyTorch Datasets...")

    # 2. Instantiate the Train Dataset
    #    (Uses global config for max_length)
    train_ds = TranslationDataset(
        dataset=train_data,
        tokenizer=tokenizer,
        max_len_src=config.MAX_SEQ_LEN,
        max_len_tgt=config.MAX_SEQ_LEN,
    )

    # 3. Instantiate the Validation Dataset
    val_ds = TranslationDataset(
        dataset=val_data,
        tokenizer=tokenizer,
        max_len_src=config.MAX_SEQ_LEN,
        max_len_tgt=config.MAX_SEQ_LEN,
    )

    # 4. Instantiate the Test Dataset
    test_ds = TranslationDataset(
        dataset=test_data,
        tokenizer=tokenizer,
        max_len_src=config.MAX_SEQ_LEN,
        max_len_tgt=config.MAX_SEQ_LEN,
    )

    print(
        f"Datasets created: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}"
    )

    return train_ds, val_ds, test_ds


def get_dataloaders(
    tokenizer: PreTrainedTokenizerFast,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    A high-level Factory function to create DataLoaders.

    This function abstracts away all the data pipeline complexity:
    - Loading/Cleaning raw data
    - Creating PyTorch Datasets
    - Instantiating the DataCollator (dynamic padding)
    - Creating DataLoaders with the correct batch size and workers

    Args:
        tokenizer: The trained tokenizer.

    Returns:
        Tuple containing (train_loader, val_loader, test_loader)
    """

    # 1. Create the Datasets (using the factory function we made earlier)
    train_ds, val_ds, test_ds = get_translation_datasets(tokenizer)

    # 2. Instantiate the Collator
    # (We need config to get PAD_TOKEN_ID)
    collator = DataCollator(pad_token_id=config.PAD_TOKEN_ID)

    print(
        f"Building DataLoaders (Batch Size: {config.BATCH_SIZE}, Workers: {config.NUM_WORKERS})..."
    )

    # 3. Create Train DataLoader
    # (Shuffle = True is CRITICAL for training)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        collate_fn=collator,
        pin_memory=True if config.DEVICE == "cuda" else False,  # (Optimization)
        prefetch_factor=2,
        persistent_workers=True,
    )

    # 4. Create Validation DataLoader
    # (Shuffle = False for reproducible validation)
    val_loader = DataLoader(
        val_ds,
        batch_size=2 * config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collator,
        pin_memory=True if config.DEVICE == "cuda" else False,
        prefetch_factor=2,
        persistent_workers=True,
    )

    # 5. Create Test DataLoader
    test_loader = DataLoader(
        test_ds,
        batch_size=2 * config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        # num_workers=config.NUM_WORKERS,
        collate_fn=collator,
        pin_memory=True if config.DEVICE == "cuda" else False,
        prefetch_factor=2,
    )

    print(f"DataLoader (train) created with {len(train_loader)} batches.")
    print(f"DataLoader (val) created with {len(val_loader)} batches.")
    print(f"DataLoader (test) created with {len(test_loader)} batches.")

    return train_loader, val_loader, test_loader
