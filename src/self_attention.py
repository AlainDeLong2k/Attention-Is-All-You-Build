import torch
from torch import Tensor
import torch.nn as nn
from jaxtyping import Float


class SingleHeadAttention(nn.Module):
    def __init__(self, embedding_dim: int, attention_dim: int) -> None:
        super().__init__()

        self.get_queries: nn.Linear = nn.Linear(
            embedding_dim, attention_dim, bias=False
        )
        self.get_keys: nn.Linear = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.get_values: nn.Linear = nn.Linear(embedding_dim, attention_dim, bias=False)

    def forward(
        self, embedded: Float[Tensor, "batch_size context_length embedding_dim"]
    ) -> Float[Tensor, "batch_size context_length attention_dim"]:
        Q = self.get_queries(embedded)  # B x T x A
        K = self.get_keys(embedded)  # B x T x A
        V = self.get_values(embedded)  # B x T x A
        B, T, A = V.shape  # B: Batch Size, T: Context Length, A: Attention Dimension

        attention_scores: Tensor = Q @ torch.transpose(K, dim0=1, dim1=2)
        attention_scores = attention_scores / (A**0.5)

        lower_triangular = torch.tril(torch.ones(T, T))
        mask = lower_triangular == 0
        attention_scores = attention_scores.masked_fill(mask, float("-inf"))
        attention_scores = nn.functional.softmax(attention_scores, dim=2)  # B x T x T

        contextual_embedded = attention_scores @ V  # B x T x A
        return torch.round(contextual_embedded, decimals=4)


if __name__ == "__main__":
    embedding_dim = 3
    attention_dim = 4
    embedded = torch.randn(2, 2, embedding_dim)

    head = SingleHeadAttention(embedding_dim, attention_dim)
    print(head.forward(embedded))
