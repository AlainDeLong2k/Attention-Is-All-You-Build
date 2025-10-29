import torch
from typing import List, Tuple


class Solution:
    def batch_loader(
        self, raw_dataset: str, context_length: int, batch_size: int
    ) -> Tuple[List[List[str]]]:
        # You must start by generating batch_size different random indices in the appropriate range
        # using a single call to torch.randint()
        torch.manual_seed(0)

        words: List[str] = raw_dataset.split(" ")
        indices = torch.randint(
            low=0, high=len(words) - context_length, size=(batch_size,)
        )

        X: List[List[str]] = []
        Y: List[List[str]] = []

        for idx in indices:
            X.append(words[idx : idx + context_length])
            Y.append(words[idx + 1 : idx + 1 + context_length])

        # return (X,)
        return X, Y


if __name__ == "__main__":
    raw_dataset = input("raw_dataset = ")
    context_length = int(input("context_length = "))
    batch_size = int(input("batch_size = "))

    sol = Solution()
    output = sol.batch_loader(raw_dataset, context_length, batch_size)
    print(output)
