import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset
from datasets import Dataset as ArrowDataset
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from config import SOS_TOKEN_ID, EOS_TOKEN_ID
from utils import create_padding_mask, create_look_ahead_mask


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

        tgt_ids = [SOS_TOKEN_ID] + tgt_encoding["input_ids"] + [EOS_TOKEN_ID]

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
        src_mask = create_padding_mask(src_ids_padded, self.pad_token_id)

        # (Mask 2) Target padding mask (for Decoder MHA)
        # Shape: (B, 1, 1, T_tgt)
        tgt_padding_mask = create_padding_mask(dec_input_ids_padded, self.pad_token_id)

        # (Mask 3) Target look-ahead mask (for Decoder MHA)
        # Shape: (1, 1, T_tgt, T_tgt)
        look_ahead_mask = create_look_ahead_mask(T_tgt)

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
