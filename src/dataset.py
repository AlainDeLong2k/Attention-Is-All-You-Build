import torch


class CustomDataset:
    def batch_loader(
        self, raw_dataset: str, context_length: int, batch_size: int
    ) -> tuple[list[list[str]], list[list[str]]]:
        torch.manual_seed(0)

        words: list[str] = raw_dataset.split(" ")
        indices = torch.randint(
            low=0, high=len(words) - context_length, size=(batch_size,)
        )

        X: list[list[str]] = []
        Y: list[list[str]] = []

        for idx in indices:
            X.append(words[idx : idx + context_length])
            Y.append(words[idx + 1 : idx + 1 + context_length])

        return X, Y


if __name__ == "__main__":
    raw_dataset = "Hello darkness my old friend"
    context_length = 3
    batch_size = 2

    dataset = CustomDataset()
    output = dataset.batch_loader(raw_dataset, context_length, batch_size)
    print(output)
