import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from src.model import Transformer


TGT_VOCAB_SIZE: int = 32_000


def train_one_epoch(
    model: Transformer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
) -> float:
    """
    Runs a single training epoch.

    Args:
        model: The Transformer model.
        dataloader: The training DataLoader.
        optimizer: The optimizer.
        criterion: The loss function (e.g., CrossEntropyLoss).
        device: The device to run on (e.g., 'cuda').

    Returns:
        The average training loss for the epoch.
    """

    # Set model to training mode
    # This enables dropout, etc.
    model.train()

    total_loss = 0.0

    # Use tqdm for a progress bar
    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch in progress_bar:
        # 1. Move batch to device (GPU)
        # We define a helper for this
        batch_gpu = {
            k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
        }

        # 2. Zero gradients before forward pass
        optimizer.zero_grad()

        # 3. Forward pass
        # Get inputs for the model (as defined in Transformer.forward)
        logits = model(
            src=batch_gpu["src_ids"],
            tgt=batch_gpu["tgt_input_ids"],
            src_mask=batch_gpu["src_mask"],
            tgt_mask=batch_gpu["tgt_mask"],
        )  # Shape: (B, T_tgt, vocab_size)

        # 4. Calculate loss
        # CrossEntropyLoss expects (N, C) and (N,)
        # We must reshape logits and labels
        # Logits: (B, T_tgt, C) -> (B * T_tgt, C)
        # Labels: (B, T_tgt) -> (B * T_tgt)
        loss = criterion(logits.view(-1, TGT_VOCAB_SIZE), batch_gpu["labels"].view(-1))

        # 5. Backward pass (compute gradients)
        loss.backward()

        # 6. Gradient Clipping (from paper)
        # Helps prevent exploding gradients. '1.0' is a common value.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 7. Update weights
        optimizer.step()
        scheduler.step()

        # 8. Update stats
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    # Return average loss for the epoch
    return total_loss / len(dataloader)


def validate_one_epoch(
    model: Transformer,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Runs a single validation epoch.

    Args:
        model: The Transformer model.
        dataloader: The validation DataLoader.
        criterion: The loss function (e.g., CrossEntropyLoss).
        device: The device to run on (e.g., 'cuda').

    Returns:
        The average validation loss for the epoch.
    """

    # Set model to evaluation mode
    # This disables dropout.
    model.eval()

    total_loss = 0.0

    # Use tqdm for a progress bar
    progress_bar = tqdm(dataloader, desc="Validating", leave=False)

    # Disable gradient computation
    # This saves VRAM and speeds up inference.
    with torch.no_grad():
        for batch in progress_bar:
            # 1. Move batch to device (GPU)
            batch_gpu = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }

            # 2. Forward pass
            logits = model(
                src=batch_gpu["src_ids"],
                tgt=batch_gpu["tgt_input_ids"],
                src_mask=batch_gpu["src_mask"],
                tgt_mask=batch_gpu["tgt_mask"],
            )  # Shape: (B, T_tgt, vocab_size)

            # 3. Calculate loss
            # (Use the same reshaping as in training for consistency)
            loss = criterion(
                logits.view(-1, TGT_VOCAB_SIZE), batch_gpu["labels"].view(-1)
            )

            # 4. Update stats
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

    # Return average loss for the epoch
    return total_loss / len(dataloader)
