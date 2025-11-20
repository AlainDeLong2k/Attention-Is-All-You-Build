import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from tqdm.auto import tqdm
from safetensors.torch import save_model, load_model
import config
from src import dataset, model, engine, callbacks, utils

# Import các thư viện tracking (WandB/Tensorboard) nếu dùng


def main():
    """
    Main training script for Transformer model.
    """

    # Step 1: Set random seed for reproducibility
    utils.seed_everything(seed=42)

    # Select device (CPU or GPU)
    device = config.DEVICE

    # Initialize logger (WandB, TensorBoard, or standard logging)
    # TODO: Set up logger for experiment tracking

    # Step 2: Prepare data
    # Load tokenizer from file or create a new one
    # tokenizer = None  # TODO
    tokenizer = utils.load_tokenizer(config.TOKENIZER_PATH)

    # Load and clean datasets, return train/val/test splits
    # cleaned_datasets = None  # TODO

    # Create DataLoader objects for training and validation
    # train_loader = None  # TODO
    # val_loader = None  # TODO
    # Test loader is usually used after training
    train_loader, val_loader, test_loader = dataset.get_dataloaders(tokenizer)

    # Step 3: Initialize model
    # Set up model configuration with hyperparameters
    # model_config = None  # TODO

    # Create Transformer model instance
    # Move model to selected device
    # # TODO: transformer.to(device)
    print("Instantiating the Transformer model...")
    transformer_model = model.Transformer(
        src_vocab_size=tokenizer.vocab_size,
        tgt_vocab_size=tokenizer.vocab_size,
        d_model=config.D_MODEL,
        n_heads=config.N_HEADS,
        n_layers=config.N_LAYERS,
        d_ff=config.D_FF,
        dropout=config.DROPOUT,
        max_seq_len=config.MAX_SEQ_LEN,
    ).to(config.DEVICE)

    print(
        f"Total model parameters: {sum(p.numel() for p in transformer_model.parameters() if p.requires_grad):,}"
    )

    # Optionally compile model for performance (PyTorch 2.0+)
    # if torch.__version__ >= "2.0":
    #     transformer = torch.compile(transformer)

    # Step 4: Set up training components
    # Initialize optimizer (AdamW recommended for Transformer)
    # optimizer = None  # TODO
    optimizer = AdamW(
        transformer_model.parameters(),
        lr=config.LEARNING_RATE,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=0.01,
    )

    # # Define loss function (e.g., CrossEntropyLoss)
    # criterion = None  # TODO
    criterion = nn.CrossEntropyLoss(ignore_index=config.PAD_TOKEN_ID)

    # # Optionally set up learning rate scheduler
    # scheduler = None  # TODO
    num_training_steps = len(train_loader) * config.EPOCHS
    num_warmup_steps = int(0.1 * num_training_steps)
    # print(num_training_steps, num_warmup_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Track best validation loss for checkpointing
    # best_val_loss = float("inf")

    # Setup callbacks
    early_stopper = callbacks.EarlyStopping(patience=5, min_delta=1e-4)

    # Create CUDA Event timers for accurate GPU profiling
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )

    # (Best Practice) Use tqdm for the outer epoch loop
    epoch_progress_bar = tqdm(range(config.EPOCHS), desc="Total Epochs")

    # Step 5: Training loop
    # print("Start Training...")

    print(f"--- Starting DRY RUN ---")
    print(f"Dataset: {config.NUM_SAMPLES_TO_USE} samples")
    print(
        f"Model: d_model={config.D_MODEL}, n_layers={config.N_LAYERS}, n_heads={config.N_HEADS}"
    )
    print(f"Device: {config.DEVICE}")

    # We create lists to store the loss history
    train_loss_history = []
    val_loss_history = []

    for epoch in epoch_progress_bar:
        # --- (Best Practice) Record start event ---
        starter.record()

        # --- 1. Run Training ---
        avg_train_loss = engine.train_one_epoch(
            transformer_model,
            train_loader,
            optimizer,
            criterion,
            scheduler,
            config.DEVICE,
        )

        # --- 2. Run Validation ---
        avg_val_loss = engine.validate_one_epoch(
            transformer_model,
            val_loader,
            criterion,
            config.DEVICE,
        )

        # --- (Best Practice) Record end event and synchronize ---
        ender.record()
        torch.cuda.synchronize()  # (Wait for GPU to finish all tasks)

        # (Best Practice) Get elapsed time from GPU events (in milliseconds)
        epoch_duration_ms = starter.elapsed_time(ender)
        epoch_duration = epoch_duration_ms / 1000.0  # Convert to seconds

        # --- 3. (Best Practice) Log the results ---
        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)

        # Update the main progress bar's description
        epoch_progress_bar.set_postfix(
            train_loss=f"{avg_train_loss:.4f}", val_loss=f"{avg_val_loss:.4f}"
        )

        print(
            f"\n[EPOCH {epoch+1}/{config.EPOCHS}] "
            f"Time: {epoch_duration:.2f}s | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}"
        )

        # Early stopping update
        early_stopper.step(avg_val_loss)

        # Save best checkpoint
        if avg_val_loss == early_stopper.best_loss:
            # torch.save(transformer_model.state_dict(), config.CHECKPOINT_PATH)
            save_model(transformer_model, filename=str(config.CHECKPOINT_PATH))

        # If stopping
        if early_stopper.should_stop:
            print("Training stopped early.")
            break

    print("\n🎉 DRY RUN COMPLETE! 🎉")

    # for epoch in range(config.EPOCHS):
    #     # Training phase
    #     train_loss = None  # TODO: Call engine.train_one_epoch

    #     # Validation phase
    #     val_loss = None  # TODO: Call engine.validate_one_epoch

    #     # Update learning rate scheduler if used
    #     # if scheduler: scheduler.step(val_loss)

    #     # Log metrics and print results
    #     print(f"Epoch {epoch}: Train Loss {train_loss} | Val Loss {val_loss}")
    #     # TODO: Log metrics to logger

    #     # Save model checkpoint if validation loss improves
    #     if val_loss is not None and val_loss < best_val_loss:
    #         best_val_loss = val_loss
    #         # TODO: torch.save(transformer.state_dict(), config.MODEL_SAVE_PATH)

    #     # Check early stopping condition
    #     # TODO: Implement early stopping logic

    # # Step 6: Final evaluation (optional)
    # print("Training Done. Starting Test Evaluation...")
    # # TODO: Evaluate/test model on test set


if __name__ == "__main__":
    main()

    # tokenizer = utils.load_tokenizer(config.TOKENIZER_PATH)
    # print(
    #     tokenizer.pad_token_id,
    #     tokenizer.unk_token_id,
    #     tokenizer.bos_token_id,
    #     tokenizer.eos_token_id,
    # )

    # train_loader, val_loader, test_loader = dataset.get_dataloaders(tokenizer)
