import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader
from jaxtyping import Bool, Int
from tqdm.auto import tqdm
from src import model, utils


TGT_VOCAB_SIZE: int = 32_000


def train_one_epoch(
    model: model.Transformer,
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
    model: model.Transformer,
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


def greedy_decode_sentence(
    model: model.Transformer,
    src: Int[Tensor, "1 T_src"],  # Input: one sentence
    src_mask: Bool[Tensor, "1 1 1 T_src"],
    max_len: int,
    sos_token_id: int,
    eos_token_id: int,
    device: torch.device,
) -> Int[Tensor, "1 T_out"]:
    """
    Performs greedy decoding for a single sentence.
    This is an autoregressive process (token by token).

    Args:
        model: The trained Transformer model (already on device).
        src: The source token IDs (e.g., English).
        src_mask: The padding mask for the source.
        max_len: The maximum length to generate.
        sos_token_id: The ID for [SOS] token.
        eos_token_id: The ID for [EOS] token.
        device: The device to run on.

    Returns:
        Tensor: The generated target token IDs (e.g., Vietnamese).
    """

    # Set model to eval mode (disables dropout)
    model.eval()

    # No gradients needed
    with torch.no_grad():

        # --- 1. Encode the source *once* ---
        # (B, T_src) -> (B, T_src, D)
        src_embedded = model.src_embed(src)
        src_with_pos = model.pos_enc(src_embedded)
        enc_output: Tensor = model.encoder(src_with_pos, src_mask)

        # --- 2. Initialize the Decoder input ---
        # Start with the [SOS] token. Shape: (1, 1)
        decoder_input: Tensor = torch.tensor(
            [[sos_token_id]], dtype=torch.long, device=device
        )  # Shape: (B=1, T_tgt=1)

        # --- 3. Autoregressive Loop ---
        for _ in range(max_len - 1):  # (Max length - 1, since we have [SOS])

            # --- a. Get Target Embedding + Position ---
            # (B, T_tgt) -> (B, T_tgt, D)
            tgt_embedded = model.tgt_embed(decoder_input)
            tgt_with_pos = model.pos_enc(tgt_embedded)

            # --- b. Create Target Mask (Causal) ---
            # We must re-create the mask every loop,
            # as T_tgt (decoder_input.size(1)) is growing.
            # Shape: (1, 1, T_tgt, T_tgt)
            T_tgt = decoder_input.size(1)
            tgt_mask = utils.create_look_ahead_mask(T_tgt).to(device)

            # --- c. Run Decoder and Generator ---
            # (B, T_tgt, D)
            dec_output: Tensor = model.decoder(
                tgt_with_pos, enc_output, src_mask, tgt_mask
            )
            # (B, T_tgt, vocab_size)
            logits: Tensor = model.generator(dec_output)

            # --- d. Get the *last* token's logits ---
            # (B, T_tgt, vocab_size) -> (B, vocab_size)
            last_token_logits = logits[:, -1, :]

            # --- e. Greedy Search (get highest prob. token) ---
            # (B, vocab_size) -> (B, 1)
            next_token: Tensor = torch.argmax(last_token_logits, dim=-1).unsqueeze(-1)

            # --- f. Append the new token ---
            # (B, T_tgt) + (B, 1) -> (B, T_tgt + 1)
            decoder_input = torch.cat([decoder_input, next_token], dim=1)

            # --- g. Check for [EOS] ---
            # If the *last* token we added is [EOS], stop generating.
            if next_token.item() == eos_token_id:
                break

        return decoder_input.squeeze(0)  # Return shape (T_out)
