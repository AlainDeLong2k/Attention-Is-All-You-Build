from torch import Tensor
import torch.nn as nn
from jaxtyping import Bool, Float
import math


class MultiHeadAttention(nn.Module):
    """
    Terminology (jaxtyping):
        B: batch_size
        T_q: target sequence length (query)
        T_k: source sequence length (key/value)
        D: d_model (model dimension)
        H: n_heads (number of heads)
        d_k: dimension of each head (d_model / n_heads)
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model: int = d_model
        self.n_heads: int = n_heads
        self.d_k: int = d_model // n_heads

        self.w_q: nn.Linear = nn.Linear(d_model, d_model, bias=False)
        self.w_k: nn.Linear = nn.Linear(d_model, d_model, bias=False)
        self.w_v: nn.Linear = nn.Linear(d_model, d_model, bias=False)
        self.w_o: nn.Linear = nn.Linear(d_model, d_model, bias=False)

        self.attention_weights: Tensor | None = None

    @staticmethod
    def attention(
        query: Float[Tensor, "B H T_q d_k"],
        key: Float[Tensor, "B H T_k d_k"],
        value: Float[Tensor, "B H T_k d_k"],
        mask: Bool[Tensor, "... 1 T_q T_k"] | None,
    ) -> tuple[Float[Tensor, "B H T_q d_k"], Float[Tensor, "B H T_q T_k"]]:
        """
        Static method for Scaled Dot-Product Attention calculation.
        This is pure, stateless logic, making it easy to test.
        (Ref: "Attention Is All You Need", Equation 1)

        Args:
            query (Tensor): Query tensor
            key (Tensor): Key tensor
            value (Tensor): Value tensor
            mask (Tensor | None): Optional mask (for padding or look-ahead).

        Returns:
            tuple[Tensor, Tensor]:
                - context_vector: The output of the attention mechanism.
                - attention_weights: The softmax-normalized attention weights.
        """

        d_k: int = query.shape[-1]

        # (B, H, T_q, d_k) @ (B, H, d_k, T_k) -> (B, H, T_q, T_k)
        attention_scores: Tensor = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(
                mask == 0, value=float("-inf")
            )

        attention_weights: Tensor = attention_scores.softmax(dim=-1)

        # (B, H, T_q, T_k) @ (B, H, T_k, d_k) -> (B, H, T_q, d_k)
        context_vector: Tensor = attention_weights @ value

        return context_vector, attention_weights

    def forward(
        self,
        q: Float[Tensor, "B T_q D"],
        k: Float[Tensor, "B T_k D"],
        v: Float[Tensor, "B T_k D"],
        mask: Bool[Tensor, "... 1 T_q T_k"] | None = None,  # Optional mask
    ) -> Float[Tensor, "B T_q D"]:
        """
        Forward pass for Multi-Head Attention.

        In Self-Attention (Encoder), q, k, and v are all the same tensor.
        In Cross-Attention (Decoder), q comes from the Decoder, while k and v
        come from the Encoder's output.

        Args:
            q: Query tensor
            k: Key tensor
            v: Value tensor
            mask: Optional mask to apply (padding or look-ahead)

        Returns:
            The context vector after multi-head attention and output projection.
        """

        B, T_q, _ = q.shape
        _, T_k, _ = k.shape  # T_k == T_v

        # (B, T, D) -> (B, T, D)
        Q: Tensor = self.w_q(q)
        K: Tensor = self.w_k(k)
        V: Tensor = self.w_v(v)

        # (B, T, D) -> (B, T, H, d_k) -> (B, H, T, d_k)
        Q = Q.view(B, T_q, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(B, T_k, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T_k, self.n_heads, self.d_k).transpose(1, 2)

        context_vector, self.attention_weights = self.attention(Q, K, V, mask)

        # (B, H, T_q, d_k) -> (B, T_q, H, d_k)
        context_vector = context_vector.transpose(1, 2).contiguous()

        # (B, T_q, H, d_k) -> (B, T_q, D)
        context_vector = context_vector.view(B, T_q, self.d_model)

        # (B, T_q, D) -> (B, T_q, D)
        output: Tensor = self.w_o(context_vector)

        return output


class PositionwiseFeedForward(nn.Module):
    """
    Implements the Position-wise Feed-Forward Network (FFN) sublayer.
    (Ref: "Attention Is All You Need", Section 3.3)

    This is a two-layer MLP (Multi-Layer Perceptron) applied independently
    to each position in the sequence.

    FFN(x) = max(0, x * W_1 + b_1) * W_2 + b_2
    (Or using ReLU activation)

    Terminology (jaxtyping):
    B: batch_size
    T: seq_len (context_length)
    D: d_model (model dimension)
    D_FF: d_ff (inner feed-forward dimension)
    """

    def __init__(self, d_model: int, d_ff: int) -> None:
        """
        Initializes the FFN.

        Args:
            d_model (int): Dimension of the model (e.g., 512).
            d_ff (int): Inner dimension of the FFN (e.g., 2048).
                        Paper suggests d_ff = 4 * d_model.
            dropout (float): Dropout probability (applied *before* the
                             second linear layer in some implementations,
                             or as part of ResidualConnection).
        """
        super().__init__()

        # (B, T, D) -> (B, T, D_FF)
        self.linear_1: nn.Linear = nn.Linear(d_model, d_ff)

        self.activation: nn.ReLU = nn.ReLU()

        # (B, T, D_FF) -> (B, T, D)
        self.linear_2: nn.Linear = nn.Linear(d_ff, d_model)

    def forward(self, x: Float[Tensor, "B T D"]) -> Float[Tensor, "B T D"]:
        """
        Forward pass for the FFN.
        Applies two linear transformations with a ReLU activation in between.

        Args:
            x: Input tensor from the previous sublayer
               (e.g., MultiHeadAttention output).

        Returns:
            Output tensor of the same shape.
        """
        # (B, T, D) -> (B, T, D_FF)
        x = self.linear_1(x)

        # (B, T, D_FF) -> (B, T, D_FF)
        x = self.activation(x)

        # (B, T, D_FF) -> (B, T, D)
        x = self.linear_2(x)

        return x
