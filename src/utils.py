import torch
from torch import Tensor
from datasets import Dataset
from jaxtyping import Bool, Int


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
