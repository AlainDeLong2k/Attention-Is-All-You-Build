import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from torchmetrics.text import BLEUScore, SacreBLEUScore
from tqdm.auto import tqdm
import config
from src import model, utils


TGT_VOCAB_SIZE: int = config.VOCAB_SIZE


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

        # 8. Update learning rate scheduler if used
        scheduler.step()

        # 9. Update stats
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


def evaluate_model(
    model: model.Transformer,
    dataloader: DataLoader,
    tokenizer: PreTrainedTokenizerFast,
    device: torch.device,
) -> tuple[float, float]:
    """
    Runs final evaluation on the test set using Beam Search
    and calculates the SacreBLEU score.
    """
    print("\n--- Starting Evaluation (BLEU + SacreBLEU) ---")

    # Set model to evaluation mode
    # This disables dropout.
    model.eval()

    all_predicted_strings = []
    all_expected_strings = []

    # --- No gradients needed ---
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):

            batch_gpu = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }

            src_ids = batch_gpu["src_ids"]
            src_mask = batch_gpu["src_mask"]
            expected_ids = batch_gpu["labels"]  # (B, T_tgt) [on GPU]

            B = src_ids.size(0)

            # --- Handle 2D Expected IDs) ---
            batch_expected_strings = []

            # Convert 2D GPU Tensor -> 2D CPU List
            expected_id_lists = expected_ids.cpu().tolist()

            # Now we iterate over the CPU list
            for id_list in expected_id_lists:
                # id_list is a 1D Python list (e.g., [70, 950, 7, 3])
                # This call is now safe
                token_list = tokenizer.convert_ids_to_tokens(id_list)
                batch_expected_strings.append(
                    utils.filter_and_detokenize(token_list, skip_special=True)
                )

            # --- Generate (decode) one sentence at a time ---
            batch_predicted_strings = []
            for i in tqdm(range(B), desc="Decoding Batch", leave=False):
                src_sentence = src_ids[i].unsqueeze(0)
                src_sentence_mask = src_mask[i].unsqueeze(0)

                # (predicted_ids is 1D Tensor [T_out] on GPU)
                predicted_ids = utils.greedy_decode_sentence(
                    model,
                    src_sentence,
                    src_sentence_mask,
                    max_len=config.MAX_SEQ_LEN,
                    sos_token_id=config.SOS_TOKEN_ID,
                    eos_token_id=config.EOS_TOKEN_ID,
                    device=device,
                )

                # Convert 1D GPU Tensor -> 1D CPU List
                predicted_id_list = predicted_ids.cpu().tolist()

                # This call is now safe
                predicted_token_list = tokenizer.convert_ids_to_tokens(
                    predicted_id_list
                )

                decoded_str = utils.filter_and_detokenize(
                    predicted_token_list, skip_special=True
                )
                batch_predicted_strings.append(decoded_str)

            # --- Store strings for final metric calculation ---
            all_predicted_strings.extend(batch_predicted_strings)
            all_expected_strings.extend([[s] for s in batch_expected_strings])

    bleu_metric = BLEUScore(n_gram=4, smooth=True).to(config.DEVICE)
    sacrebleu_metric = SacreBLEUScore(
        n_gram=4, smooth=True, tokenize="intl", lowercase=False
    ).to(config.DEVICE)

    # --- 5. Calculate final score ---
    print("\nCalculating final BLEU score...")
    final_bleu = bleu_metric(all_predicted_strings, all_expected_strings)

    # print(f"\n========================================")
    # print(f"🎉 FINAL BLEU SCORE (Evaluation Set): {final_bleu.item() * 100:.4f}%")
    # print(f"========================================")

    print("\nCalculating final SacreBLEU score...")
    final_sacrebleu = sacrebleu_metric(all_predicted_strings, all_expected_strings)

    # print(f"\n========================================")
    # print(
    #     f"🎉 FINAL SacreBLEU SCORE (Evaluation Set): {final_sacrebleu.item() * 100:.4f}%"
    # )
    # print(f"========================================")

    # --- Show some examples ---
    print("\n--- Translation Examples (Pred vs Exp) ---")
    for i in range(min(5, len(all_predicted_strings))):
        print(f"  PRED: {all_predicted_strings[i]}")
        print(f"  EXP:  {all_expected_strings[i][0]}")
        print("  ---")

    return final_bleu.item() * 100, final_sacrebleu.item() * 100
