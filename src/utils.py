from pathlib import Path
import random
import re
from datetime import datetime
import numpy as np
from datasets import DatasetDict, Dataset, load_dataset
import torch
from torch import Tensor
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from jaxtyping import Bool, Int
from src import model


# Utility function to set random seed for reproducibility
def seed_everything(seed: int = 42) -> None:
    """
    Set random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_run_name(model_name: str, d_model: int) -> str:
    time_tag: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model_name}-{d_model}d-{time_tag}"


# --- Helper functions for cleaning ---
def is_valid_pair(example: dict) -> bool:
    """Check if both 'en' and 'vi' strings are non-empty."""
    translation = example.get("translation", {})
    en_text = translation.get("en", "").strip()
    vi_text = translation.get("vi", "").strip()
    return bool(en_text) and bool(vi_text)  # (Return True if both are valid)


def filter_empty(dataset: Dataset, num_proc: int) -> Dataset:
    """
    Applies the validation filter to a dataset split using
    parallel processing (via .map() or .filter()).
    """
    print(f"  Filtering empty strings from split...")
    # (We use .filter() which is highly optimized)
    original_len = len(dataset)

    filtered_dataset = dataset.filter(
        is_valid_pair, num_proc=num_proc  # (Use parallel processing from config)
    )

    new_len = len(filtered_dataset)
    print(f"  Filtered {original_len - new_len} empty/invalid pairs.")
    return filtered_dataset


# --- Dataset Loading & Splitting ---
def get_raw_data(
    dataset_path: str | Path, for_tokenizer: bool = False, num_workers: int = 8
) -> Dataset | tuple[Dataset, Dataset, Dataset]:
    """
    Load and filter dataset splits from a given path.

    Args:
        dataset_path (str | Path): Path to the dataset directory or config.
        for_tokenizer (bool): If True, return only filtered train split (for tokenizer training).
                             If False, return tuple of (train, validation, test) splits (for model training/eval).
        num_workers (int): Number of workers for parallel filtering.

    Returns:
        Dataset: Filtered train split (if for_tokenizer=True).
        tuple(Dataset, Dataset, Dataset): Filtered train, validation, test splits (if for_tokenizer=False).
    """
    print(f"Loading datasets from: {dataset_path}")
    all_splits: DatasetDict = load_dataset(path=str(dataset_path))
    print(all_splits)

    print("--- Filtering Datasets (Removing empty sentences) ---")
    train_data: Dataset = filter_empty(all_splits["train"], num_workers)
    val_data: Dataset = filter_empty(all_splits["validation"], num_workers)
    test_data: Dataset = filter_empty(all_splits["test"], num_workers)

    if for_tokenizer:
        return train_data
    else:
        return train_data, val_data, test_data


# Utility function to set random seed for reproducibility
def load_tokenizer(tokenizer_path: str | Path) -> PreTrainedTokenizerFast:
    """
    Load a trained tokenizer from file and return tokenizer object and special token ids.
    Args:
        tokenizer_path (str | Path): Path to the tokenizer JSON file.
        special_tokens (list[str], optional): List of special tokens to get ids for (e.g. ["[PAD]", "[SOS]", "[EOS]", "[UNK]"]).
    Returns:
        tokenizer (Tokenizer): Loaded tokenizer object.
        token_ids (dict): Dictionary of special token ids.
    """
    print(f"Loading tokenizer from {tokenizer_path}...")
    # tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path))
    tokenizer.pad_token = "[PAD]"
    tokenizer.unk_token = "[UNK]"
    tokenizer.bos_token = "[SOS]"  # bos = Beginning Of Sentence
    tokenizer.eos_token = "[EOS]"  # eos = End Of Sentence
    return tokenizer


def create_padding_mask(
    input_ids: Int[Tensor, "B T_k"], pad_token_id: int
) -> Bool[Tensor, "B 1 1 T_k"]:
    """
    Creates a padding mask for the attention mechanism.

    This mask identifies positions holding the <PAD> token
    and prepares a mask tensor that, when broadcasted, will mask
    these positions in the attention scores matrix (B, H, T_q, T_k).

    Args:
        input_ids (Tensor): The input token IDs. Shape (B, T_k).
        pad_token_id (int): The ID of the padding token.

    Returns:
        Tensor: A boolean mask of shape (B, 1, 1, T_k).
                'True' means "keep" (not a pad token).
                'False' means "mask out" (is a pad token).
    """

    # 1. Create the base mask
    # (input_ids != pad_token_id) will be True for real tokens, False for PAD
    # Shape: (B, T_k)
    mask: Tensor = input_ids != pad_token_id

    # 2. Add dimensions for broadcasting
    # We add a dimension for T_q (dim 1) and H (dim 2)
    # Shape: (B, T_k) -> (B, 1, T_k) -> (B, 1, 1, T_k)
    return mask.unsqueeze(1).unsqueeze(2)


def create_look_ahead_mask(seq_len: int) -> Bool[Tensor, "1 1 T_q T_q"]:
    """
    Creates a causal (look-ahead) mask for the Decoder's self-attention.

    This mask prevents positions from attending to subsequent positions.
    It's a square matrix where the upper triangle (future) is False
    and the lower triangle (past/present) is True.

    Args:
        seq_len (int): The sequence length (T_q).
        device (torch.device): The device to create the tensor on (e.g., 'cuda').

    Returns:
        Tensor: A boolean mask of shape (1, 1, T_q, T_q).
                'True' means "keep" (allowed to see).
                'False' means "mask out" (future token).
    """

    # 1. Create a square matrix of ones.
    # Shape: (T_q, T_q)
    ones = torch.ones(seq_len, seq_len)

    # 2. Get the lower triangular part (bao gồm đường chéo)
    # This sets the upper triangle (future) to 0 and keeps the rest 1.
    # Shape: (T_q, T_q)
    # Example (T_q=3):
    # [[1., 0., 0.],
    #  [1., 1., 0.],
    #  [1., 1., 1.]]
    lower_triangular: Tensor = torch.tril(ones)

    # 3. Convert to boolean and add broadcasting dimensions
    # Shape: (T_q, T_q) -> (1, 1, T_q, T_q)
    # (mask == 1) converts 1. to True, 0. to False
    return (lower_triangular == 1).unsqueeze(0).unsqueeze(0)


def greedy_decode_sentence(
    model: model.Transformer,
    src: Int[Tensor, "1 T_src"],  # Input: one sentence
    src_mask: Bool[Tensor, "1 1 1 T_src"],
    max_len: int,
    sos_token_id: int,
    eos_token_id: int,
    device: torch.device,
) -> Int[Tensor, "1 T_out"]:
    """
    Performs greedy decoding for a single sentence.
    This is an autoregressive process (token by token).

    Args:
        model: The trained Transformer model (already on device).
        src: The source token IDs (e.g., English).
        src_mask: The padding mask for the source.
        max_len: The maximum length to generate.
        sos_token_id: The ID for [SOS] token.
        eos_token_id: The ID for [EOS] token.
        device: The device to run on.

    Returns:
        Tensor: The generated target token IDs (e.g., Vietnamese).
    """

    # Set model to eval mode (disables dropout)
    model.eval()

    # No gradients needed
    with torch.no_grad():

        # --- 1. Encode the source *once* ---
        # (B, T_src) -> (B, T_src, D)
        src_embedded = model.src_embed(src)
        src_with_pos = model.pos_enc(src_embedded)
        enc_output: Tensor = model.encoder(src_with_pos, src_mask)

        # --- 2. Initialize the Decoder input ---
        # Start with the [SOS] token. Shape: (1, 1)
        decoder_input: Tensor = torch.tensor(
            [[sos_token_id]], dtype=torch.long, device=device
        )  # Shape: (B=1, T_tgt=1)

        # --- 3. Autoregressive Loop ---
        for _ in range(max_len - 1):  # (Max length - 1, since we have [SOS])

            # --- a. Get Target Embedding + Position ---
            # (B, T_tgt) -> (B, T_tgt, D)
            tgt_embedded = model.tgt_embed(decoder_input)
            tgt_with_pos = model.pos_enc(tgt_embedded)

            # --- b. Create Target Mask (Causal) ---
            # We must re-create the mask every loop,
            # as T_tgt (decoder_input.size(1)) is growing.
            # Shape: (1, 1, T_tgt, T_tgt)
            T_tgt = decoder_input.size(1)
            tgt_mask = create_look_ahead_mask(T_tgt).to(device)

            # --- c. Run Decoder and Generator ---
            # (B, T_tgt, D)
            dec_output: Tensor = model.decoder(
                tgt_with_pos, enc_output, src_mask, tgt_mask
            )
            # (B, T_tgt, vocab_size)
            logits: Tensor = model.generator(dec_output)

            # --- d. Get the *last* token's logits ---
            # (B, T_tgt, vocab_size) -> (B, vocab_size)
            last_token_logits = logits[:, -1, :]

            # --- e. Greedy Search (get highest prob. token) ---
            # (B, vocab_size) -> (B, 1)
            next_token: Tensor = torch.argmax(last_token_logits, dim=-1).unsqueeze(-1)

            # --- f. Append the new token ---
            # (B, T_tgt) + (B, 1) -> (B, T_tgt + 1)
            decoder_input = torch.cat([decoder_input, next_token], dim=1)

            # --- g. Check for [EOS] ---
            # If the *last* token we added is [EOS], stop generating.
            if next_token.item() == eos_token_id:
                break

        return decoder_input.squeeze(0)  # Return shape (T_out)


def filter_and_detokenize(token_list: list[str], skip_special: bool = True) -> str:
    """
    Manually joins tokens with a space and cleans up common
    punctuation issues caused by whitespace tokenization.
    """
    if skip_special:
        # 1. Filter out special tokens
        special_tokens = {"[PAD]", "[UNK]", "[SOS]", "[EOS]"}
        token_list = [tok for tok in token_list if tok not in special_tokens]

    # 2. Join with spaces
    detokenized_string = " ".join(token_list)

    # 3. Clean up punctuation
    # (This is a simple heuristic-based detokenizer)
    # Remove space before punctuation: "project ." -> "project."
    detokenized_string = re.sub(r'\s([.,!?\'":;])', r"\1", detokenized_string)
    # Handle contractions: "don 't" -> "don't"
    detokenized_string = re.sub(r"(\w)\s(\'\w)", r"\1\2", detokenized_string)

    return detokenized_string


# Define a high-level, production-ready
# inference function that handles all steps.
def translate(
    model: model.Transformer,
    tokenizer: PreTrainedTokenizerFast,
    sentence_en: str,
    device: torch.device,
    max_len: int,
    sos_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
) -> str:
    """
    Translates a single English sentence to Vietnamese.

    Args:
        model: The trained Transformer model.
        tokenizer: The (PreTrainedTokenizerFast) tokenizer.
        sentence_en: The raw English input string.
        device: The device to run on.
        max_len: The max sequence length (from config).
        sos_token_id: The ID for [SOS].
        eos_token_id: The ID for [EOS].
        pad_token_id: The ID for [PAD].

    Returns:
        str: The translated Vietnamese string.
    """

    # Set model to evaluation mode
    model.eval()

    # Run inference in a no-gradient context
    with torch.no_grad():

        # 1. Tokenize the source (English) sentence
        src_encoding = tokenizer(
            sentence_en,
            truncation=True,
            max_length=max_len,
            add_special_tokens=False,  # (Encoder does not need SOS/EOS)
        )

        # 2. Convert to Tensor, add Batch dimension (B=1), and move to device
        # Shape: (1, T_src)
        src_ids: Tensor = torch.tensor(
            [src_encoding["input_ids"]], dtype=torch.long
        ).to(device)

        # 3. Create the source padding mask
        # Shape: (1, 1, 1, T_src)
        src_mask: Tensor = create_padding_mask(src_ids, pad_token_id).to(device)

        # 4. Generate the target (Vietnamese) token IDs
        # (This calls the autoregressive function from Cell 16A)
        # Shape: (T_out)
        predicted_ids: Tensor = greedy_decode_sentence(
            model,
            src_ids,
            src_mask,
            max_len=max_len,
            sos_token_id=sos_token_id,
            eos_token_id=eos_token_id,
            device=device,
        )

        # 5. Detokenize (Fixing "sticky" words)

        # Convert 1D GPU Tensor -> 1D CPU List
        predicted_id_list = predicted_ids.cpu().tolist()

        # This call is safe (1D List -> List[str])
        predicted_token_list = tokenizer.convert_ids_to_tokens(predicted_id_list)

        # Use our helper (from Cell 16B) to
        # join with spaces, remove special tokens, and fix punctuation.
        result_string = filter_and_detokenize(predicted_token_list, skip_special=True)

        return result_string

    print("Inference function `translate()` defined.")
