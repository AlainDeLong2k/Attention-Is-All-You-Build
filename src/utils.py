import torch
from torch import Tensor
from jaxtyping import Bool, Int


def create_padding_mask(
    input_tensor: Int[Tensor, "B T"], pad_token_id: int
) -> Bool[Tensor, "B 1 1 T"]:
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

    mask: Tensor = input_tensor != pad_token_id

    # (B, T) -> (B, 1, 1, T) for broadcasting
    return mask.unsqueeze(1).unsqueeze(2)


def create_look_ahead_mask(seq_len: int) -> Bool[Tensor, "1 1 T T"]:
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

    # (T, T)
    lower_triangular: Tensor = torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool)
    )

    # (1, 1, T, T) for broadcasting
    return lower_triangular.unsqueeze(0).unsqueeze(0)
