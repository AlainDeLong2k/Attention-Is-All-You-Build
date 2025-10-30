import torch
from torch import Tensor
import torch.nn as nn
from jaxtyping import Float
from self_attention import SingleHeadAttention


class MultiHeadedSelfAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int, num_heads: int):
        super().__init__()

        self.heads = nn.ModuleList()

        for _ in range(num_heads):
            self.heads.append(
                SingleHeadAttention(embedding_dim, attention_dim // num_heads)
            )

    def forward(
        self, embedded: Float[Tensor, "batch_size context_length embedding_dim"]
    ) -> Float[Tensor, "batch_size context_length attention_dim"]:
        outputs = []  # each element in this list is B*T*Head_Size --> B*T*Attention_Dim

        for head in self.heads:
            outputs.append(head.forward(embedded))

        concat = torch.cat(outputs, dim=2)
        return concat


if __name__ == "__main__":
    embedding_dim = 3
    attention_dim = 4
    num_heads = 2
    embedded = torch.randn(2, 2, embedding_dim)

    multi_head = MultiHeadedSelfAttention(embedding_dim, attention_dim, num_heads)
    print(multi_head.forward(embedded))
