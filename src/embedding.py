import torch
from torch import Tensor
import torch.nn as nn
from jaxtyping import Int, Float
import math


class InputEmbeddings(nn.Module):
    """
    Implements the Input Embedding layer.

    This module converts a tensor of token IDs into a tensor of
    corresponding embedding vectors. It also scales the embeddings
    by sqrt(d_model) as mentioned in the paper ("Attention Is All You Need",
    Section 3.4).
    """

    def __init__(self, d_model: int, vocab_size: int) -> None:
        """
        Initializes the InputEmbedding layer.

        Args:
            d_model (int): The dimension of the embedding vector (D).
            vocab_size (int): The size of the vocabulary.
        """
        super().__init__()

        self.d_model: int = d_model
        self.vocab_size: int = vocab_size

        self.token_emb: nn.Embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: Int[Tensor, "B T"]) -> Float[Tensor, "B T D"]:
        """
        Forward pass for the InputEmbeddings.

        Args:
            x (Tensor): Input tensor of token IDs. Shape (B, T). B: batch_size, T: seq_len

        Returns:
            Tensor: The corresponding embedding vectors, scaled by sqrt(d_model).
                    Shape (B, T, D).
        """
        # (B, T) -> (B, T, D)
        embeddings = self.token_emb(x)

        return embeddings * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Implements the fixed (sin/cos) Positional Encoding module.
    (Ref: "Attention Is All You Need", Section 3.5)

    This module generates a tensor of positional encodings that are
    added to the input embeddings. It also applies dropout to the
    sum of the embeddings and the positional encodings.
    """

    def __init__(self, d_model: int, max_seq_len: int, dropout: float = 0.1) -> None:
        """
        Initializes the PositionalEncoding module.

        Args:
            d_model (int): The dimension of the model (D).
            max_seq_len (int): The maximum sequence length (T_max) to pre-compute.
            dropout (float): Dropout probability.
        """
        super().__init__()

        self.dropout: nn.Dropout = nn.Dropout(p=dropout)

        position: Tensor = torch.arange(max_seq_len).unsqueeze(1).float()

        div_term: Tensor = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model)
        )

        # (T_max, D)
        pe: Tensor = torch.zeros(max_seq_len, d_model)

        pe[:, 0::2] = torch.sin(position * div_term)

        pe[:, 1::2] = torch.cos(position * div_term)

        # (T_max D) -> (1, T_max, D)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: Float[Tensor, "B T D"]) -> Float[Tensor, "B T D"]:
        """
        Adds positional encoding to the input embeddings and applies dropout.

        Args:
            x (Tensor): Input tensor (token embeddings, already scaled).
                        Shape (B, T, D).

        Returns:
            Tensor: Output tensor with positional information and dropout.
                    Shape (B, T, D).
        """
        x = x + self.pe[:, : x.size(1), :]

        return self.dropout(x)
