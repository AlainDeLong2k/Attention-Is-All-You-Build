from torch import Tensor
import torch.nn as nn
from typing import Callable
from jaxtyping import Bool, Float
from layers import MultiHeadAttention, PositionwiseFeedForward


class ResidualConnection(nn.Module):
    """
    Implements the (Pre-LN) Residual Connection module, which wraps a sublayer
    (like MultiHeadAttention or FFN) with LayerNormalization and Dropout.

    This is the modern "best practice" used in models like GPT-2, which is
    more stable than the original Post-LN design in "Attention Is All You Need".

    Architecture: x = x + Dropout(Sublayer(LayerNorm(x)))
    """

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        """
        Initializes the Residual Connection.

        Args:
            d_model (int): The dimension of the model (D).
            dropout (float): Dropout probability to apply to the sublayer output.
        """
        super().__init__()

        self.dropout: nn.Dropout = nn.Dropout(dropout)

        self.norm: nn.LayerNorm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: Float[Tensor, "B T D"],
        sublayer: Callable[[Float[Tensor, "B T D"]], Float[Tensor, "B T D"]],
    ) -> Float[Tensor, "B T D"]:
        """
        Forward pass for the Residual Connection.

        Args:
            x (Tensor): The input tensor from the previous layer.
            sublayer (Callable): The sublayer module (e.g., MHA or FFN)
                                 to apply the connection to.

        Returns:
            Tensor: The output tensor after the residual connection.
        """

        x_normed = self.norm(x)

        sublayer_output = sublayer(x_normed)

        dropout_output = self.dropout(sublayer_output)

        return x + dropout_output


class EncoderLayer(nn.Module):
    """
    Implements one single Encoder Layer (or "Block") of the Transformer Encoder.

    An Encoder Layer consists of two main sublayers:
    1. A Multi-Head Self-Attention mechanism (MHA).
    2. A Position-wise Feed-Forward Network (FFN).

    Each sublayer is wrapped by a ResidualConnection (which includes
    Pre-LayerNormalization and Dropout).

    Architecture:
    x -> Residual_1(x, MHA) -> x'
    x' -> Residual_2(x', FFN) -> output
    """

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1
    ) -> None:
        """
        Initializes the Encoder Layer.

        Args:
            d_model (int): The dimension of the model (D).
            n_heads (int): The number of attention heads (H).
            d_ff (int): The inner dimension of the Feed-Forward Network (D_FF).
            dropout (float): The dropout rate for the residual connections.
        """
        super().__init__()

        self.self_attn: MultiHeadAttention = MultiHeadAttention(d_model, n_heads)

        self.feed_forward: PositionwiseFeedForward = PositionwiseFeedForward(
            d_model, d_ff
        )

        self.residual_1: ResidualConnection = ResidualConnection(d_model, dropout)
        self.residual_2: ResidualConnection = ResidualConnection(d_model, dropout)

    def forward(
        self, x: Float[Tensor, "B T D"], src_mask: Bool[Tensor, "B 1 1 T_k"]
    ) -> Float[Tensor, "B T D"]:
        """
        Forward pass for the Encoder Layer.

        Args:
            x (Tensor): Input tensor from the previous layer or embedding.
            src_mask (Tensor): The padding mask for the source sentence.
                               Shape (B, 1, 1, T_k) allows broadcasting
                               to (B, H, T_q, T_k).

        Returns:
            Tensor: The output tensor of the Encoder Layer.
        """
        x = self.residual_1(
            x,
            lambda x_normed: self.self_attn(
                q=x_normed, k=x_normed, v=x_normed, mask=src_mask
            ),
        )

        x = self.residual_2(x, self.feed_forward)

        return x


class Encoder(nn.Module):
    """
    Implements the full Transformer Encoder, which is a stack of N
    identical EncoderLayers.

    This module takes the input embeddings + positional encodings and
    processes them through N layers of self-attention and FFNs.

    (Best Practice: Uses Pre-LN, so a final LayerNorm is applied
    at the *end* of the stack, before passing to the Decoder).
    """

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, n_layers: int, dropout: float = 0.1
    ) -> None:
        """
        Initializes the Encoder stack.

        Args:
            d_model (int): The dimension of the model (D).
            n_heads (int): The number of attention heads (H).
            d_ff (int): The inner dimension of the Feed-Forward Network (D_FF).
            n_layers (int): The number of EncoderLayer blocks to stack (N).
            dropout (float): The dropout rate for the residual connections.
        """
        super().__init__()

        self.layers: nn.ModuleList = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

        self.norm: nn.LayerNorm = nn.LayerNorm(d_model)

    def forward(
        self, x: Float[Tensor, "B T D"], src_mask: Bool[Tensor, "B 1 1 T"]
    ) -> Float[Tensor, "B T D"]:
        """
        Forward pass for the entire Encoder stack.

        Args:
            x (Tensor): Input tensor (usually token embeddings + pos encodings).
            src_mask (Tensor): The padding mask for the source sentence.

        Returns:
            Tensor: The output of the final Encoder layer (the "context"
                    or "memory" for the Decoder).
        """

        for layer in self.layers:
            x = layer(x, src_mask)

        x = self.norm(x)

        return x


class DecoderLayer(nn.Module):
    """
    Implements one single Decoder Layer (or "Block") of the Transformer Decoder.

    A Decoder Layer consists of three main sublayers:
    1. A Masked Multi-Head Self-Attention mechanism (MHA).
    2. A Multi-Head Cross-Attention mechanism (MHA).
    3. A Position-wise Feed-Forward Network (FFN).

    Each sublayer is wrapped by a ResidualConnection (Pre-LN and Dropout).

    Architecture:
    x -> Residual_1(x, Masked_MHA) -> x'
    x' -> Residual_2(x', Cross_MHA, enc_output) -> x''
    x'' -> Residual_3(x'', FFN) -> output
    """

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1
    ) -> None:
        """
        Initializes the Decoder Layer.

        Args:
            d_model (int): The dimension of the model (D).
            n_heads (int): The number of attention heads (H).
            d_ff (int): The inner dimension of the Feed-Forward Network (D_FF).
            dropout (float): The dropout rate for the residual connections.
        """
        super().__init__()

        self.self_attn: MultiHeadAttention = MultiHeadAttention(d_model, n_heads)

        self.cross_attn: MultiHeadAttention = MultiHeadAttention(d_model, n_heads)

        self.feed_forward: PositionwiseFeedForward = PositionwiseFeedForward(
            d_model, d_ff
        )

        self.residual_1: ResidualConnection = ResidualConnection(d_model, dropout)
        self.residual_2: ResidualConnection = ResidualConnection(d_model, dropout)
        self.residual_3: ResidualConnection = ResidualConnection(d_model, dropout)

    def forward(
        self,
        x: Float[Tensor, "B T_tgt D"],
        enc_output: Float[Tensor, "B T_src D"],
        src_mask: Bool[Tensor, "B 1 1 T_src"],
        tgt_mask: Bool[Tensor, "B 1 1 T_tgt"],
    ) -> Float[Tensor, "B T_tgt D"]:
        """
        Forward pass for the Decoder Layer.

        Args:
            x (Tensor): Input tensor from the previous decoder layer.
            enc_output (Tensor): The output tensor from the Encoder (K, V).
            src_mask (Tensor): The padding mask for the source (Encoder) input.
            tgt_mask (Tensor): The combined look-ahead and padding mask
                               for the target (Decoder) input.

        Returns:
            Tensor: The output tensor of the Decoder Layer.
        """
        x = self.residual_1(
            x,
            lambda x_normed: self.self_attn(
                q=x_normed, k=x_normed, v=x_normed, mask=tgt_mask
            ),
        )

        x = self.residual_2(
            x,
            lambda x_normed: self.cross_attn(
                q=x_normed, k=enc_output, v=enc_output, mask=src_mask
            ),
        )

        x = self.residual_3(x, self.feed_forward)

        return x


class Decoder(nn.Module):
    """
    Implements the full Transformer Decoder, which is a stack of N
    identical DecoderLayers.

    This module takes the target embeddings + positional encodings and
    processes them through N layers of masked self-attention,
    cross-attention, and FFNs.

    (Best Practice: Uses Pre-LN, so a final LayerNorm is applied
    at the *end* of the stack, before passing to the final Generator).
    """

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, n_layers: int, dropout: float = 0.1
    ) -> None:
        """
        Initializes the Decoder stack.

        Args:
            d_model (int): The dimension of the model (D).
            n_heads (int): The number of attention heads (H).
            d_ff (int): The inner dimension of the Feed-Forward Network (D_FF).
            n_layers (int): The number of DecoderLayer blocks to stack (N).
            dropout (float): The dropout rate for the residual connections.
        """
        super().__init__()

        self.layers: nn.ModuleList = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

        self.norm: nn.LayerNorm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: Float[Tensor, "B T_tgt D"],
        enc_output: Float[Tensor, "B T_src D"],
        src_mask: Bool[Tensor, "B 1 1 T_src"],
        tgt_mask: Bool[Tensor, "1 1 T_tgt T_tgt"],
    ) -> Float[Tensor, "B T_tgt D"]:
        """
        Forward pass for the entire Decoder stack.

        Args:
            x (Tensor): Input tensor for the target (embeddings + pos enc).
            enc_output (Tensor): The output from the Encoder (K, V for cross-attn).
            src_mask (Tensor): Padding mask for the source (Encoder) sequence.
            tgt_mask (Tensor): Combined mask for the target (Decoder) sequence.

        Returns:
            Tensor: The output of the final Decoder layer, ready for the
                    final projection (Generator).
        """

        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)

        x = self.norm(x)

        return x
