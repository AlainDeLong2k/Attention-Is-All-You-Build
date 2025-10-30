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

    # class SingleHeadAttention(nn.Module):
    #     def __init__(self, embedding_dim: int, attention_dim: int):
    #         super().__init__()
    #         torch.manual_seed(0)
    #         self.key_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
    #         self.query_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
    #         self.value_gen = nn.Linear(embedding_dim, attention_dim, bias=False)

    #     def forward(self, embedded: TensorType[float]) -> TensorType[float]:
    #         k = self.key_gen(embedded)
    #         q = self.query_gen(embedded)
    #         v = self.value_gen(embedded)

    #         scores = q @ torch.transpose(k, 1, 2)  # @ is the same as torch.matmul()
    #         context_length, attention_dim = k.shape[1], k.shape[2]
    #         scores = scores / (attention_dim**0.5)

    #         lower_triangular = torch.tril(torch.ones(context_length, context_length))
    #         mask = lower_triangular == 0
    #         scores = scores.masked_fill(mask, float("-inf"))
    #         scores = nn.functional.softmax(scores, dim=2)

    #         return scores @ v


if __name__ == "__main__":
    embedding_dim = 3
    attention_dim = 4
    num_heads = 2
    embedded = torch.randn(2, 2, embedding_dim)

    multi_head = MultiHeadedSelfAttention(embedding_dim, attention_dim, num_heads)
    print(multi_head.forward(embedded))
