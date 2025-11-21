import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
import wandb
from tqdm.auto import tqdm
from safetensors.torch import save_model
import config
from src import dataset, model, engine, callbacks, utils


def main():
    """
    Main training script for Transformer model.
    """

    # Step 1: Set random seed for reproducibility
    utils.seed_everything(seed=42)

    # Initialize logger (WandB, TensorBoard, or standard logging)
    # TODO: Set up logger for experiment tracking
    config_dict = {
        k: v for k, v in vars(config).items() if k.isupper() and not k.startswith("_")
    }

    run = wandb.init(
        entity="alaindelong-hcmut",
        project="Attention Is All You Build",
        name=utils.make_run_name("transformer_en_vi_iwslt", config.D_MODEL),
        config=config_dict,
        job_type="train",
    )

    # Step 2: Prepare data
    # Load tokenizer from file or create a new one
    # tokenizer = None  # TODO
    tokenizer = utils.load_tokenizer(config.TOKENIZER_PATH)

    # Create DataLoader objects for training and validation
    # Test loader is usually used after training
    train_loader, val_loader, test_loader = dataset.get_dataloaders(tokenizer)

    # Step 3: Initialize model
    # Create Transformer model instance
    # Move model to selected device
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

    # Step 4: Set up training components
    # Initialize optimizer (AdamW recommended for Transformer)
    optimizer = AdamW(
        transformer_model.parameters(),
        lr=config.LEARNING_RATE,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=0.01,
    )

    # Define loss function (e.g., CrossEntropyLoss)
    criterion = nn.CrossEntropyLoss(
        ignore_index=config.PAD_TOKEN_ID, label_smoothing=0.1
    ).to(config.DEVICE)

    # Optionally set up learning rate scheduler
    num_training_steps = len(train_loader) * config.EPOCHS
    num_warmup_steps = int(0.1 * num_training_steps)
    # print(num_training_steps, num_warmup_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Setup callbacks (Early Stopping)
    # Track best validation loss for checkpointing
    early_stopper = callbacks.EarlyStopping(patience=5, min_delta=1e-4)

    # Create CUDA Event timers for accurate GPU profiling
    # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    # Use tqdm for the outer epoch loop
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
        # --- Record start event ---
        # starter.record()
        start = time.perf_counter()

        # --- Training phase ---
        avg_train_loss = engine.train_one_epoch(
            transformer_model,
            train_loader,
            optimizer,
            criterion,
            scheduler,
            config.DEVICE,
            run,
        )

        # --- Validation phase ---
        avg_val_loss = engine.validate_one_epoch(
            transformer_model,
            val_loader,
            criterion,
            config.DEVICE,
        )

        # --- Record end event and synchronize ---
        # ender.record()
        end = time.perf_counter()
        torch.cuda.synchronize()  # (Wait for GPU to finish all tasks)

        # Get elapsed time from GPU events (in milliseconds)
        # epoch_duration_ms = starter.elapsed_time(ender)
        # epoch_duration = epoch_duration_ms / 1000.0  # Convert to seconds
        epoch_duration = end - start

        # --- 3. Log the results ---
        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)

        # Update the main progress bar's description
        epoch_progress_bar.set_postfix(
            train_loss=f"{avg_train_loss:.4f}", val_loss=f"{avg_val_loss:.4f}"
        )

        # Log metrics and print results
        run.log(
            {"epoch": epoch + 1, "train/loss": avg_train_loss, "val/loss": avg_val_loss}
        )

        print(
            f"\n[EPOCH {epoch+1}/{config.EPOCHS}] "
            f"Time: {epoch_duration:.2f}s | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}"
        )

        # Early stopping update
        early_stopper.step(avg_val_loss)

        # Save model checkpoint if validation loss improves
        if avg_val_loss == early_stopper.best_loss:
            # torch.save(transformer_model.state_dict(), config.CHECKPOINT_PATH)
            save_model(transformer_model, filename=str(config.CHECKPOINT_PATH))

        # Check early stopping condition
        if early_stopper.should_stop:
            print("Training stopped early.")
            break

    print("\n🎉 DRY RUN COMPLETE! 🎉")

    # --- STEP 6: Final Evaluation ---

    # 1. Load the BEST saved model weights
    # (We don't want the last epoch's weights, we want the best validation weights)
    try:
        transformer_model = model.load_trained_model(
            config, config.CHECKPOINT_PATH, config.DEVICE
        )

    except Exception as e:
        print(f"⚠️ Could not load checkpoint: {e}")
        print("Using current model weights instead.")

    columns = ["Target", "Prediction"]
    translation_table = wandb.Table(columns=columns)

    # 2. Run Evaluation on TEST set
    test_bleu, test_sacrebleu = engine.evaluate_model(
        model=transformer_model,
        dataloader=test_loader,
        tokenizer=tokenizer,
        device=config.DEVICE,
        table=translation_table,
    )

    print(
        f"\n✅ Project Completed. Final Test BLEU Score: {test_bleu:.4f}% | Final Test SacreBLEU Score: {test_sacrebleu:.4f}%"
    )

    run.log(
        {
            "test/bleu": test_bleu,
            "test/sacre_bleu": test_sacrebleu,
            "test/translations": translation_table,
        }
    )

    save_model(transformer_model, filename=str(config.MODEL_SAVE_PATH))

    # Log model
    model_artifact = wandb.Artifact(name="transformer_en_vi_iwslt", type="model")
    model_artifact.add_file(
        local_path=str(config.MODEL_SAVE_PATH), name=config.MODEL_NAME
    )
    run.log_artifact(model_artifact, aliases=["best", f"{config.D_MODEL}d"])

    run.finish()


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

    # transformer_model = model.load_trained_model(
    #     config, config.MODEL_SAVE_PATH, config.DEVICE
    # )

    # # 2. Run Evaluation on TEST set
    # test_bleu, test_sacrebleu = engine.evaluate_model(
    #     model=transformer_model,
    #     dataloader=test_loader,  # Use Test Loader here
    #     tokenizer=tokenizer,
    #     device=config.DEVICE,
    # )

    # print(
    #     f"\n✅ Project Completed. Final Test BLEU: {test_bleu:.4f} | Final Test SacreBLEU: {test_sacrebleu:.4f}"
    # )

    # config_dict = {
    #     k: v for k, v in vars(config).items() if k.isupper() and not k.startswith("_")
    # }

    # run = wandb.init(
    #     entity="alaindelong-hcmut",
    #     project="Attention Is All You Build",
    #     config=config_dict,
    #     job_type="train",
    # )

    # # Log model
    # model_artifact = wandb.Artifact(name="transformer_en_vi_iwslt", type="model")
    # model_artifact.add_file(
    #     local_path=str(config.MODEL_SAVE_PATH), name=config.MODEL_NAME
    # )
    # run.log_artifact(model_artifact, aliases=["best", f"{config.D_MODEL}d"])

    # # Log tokenizer
    # tokenizer_artifact = wandb.Artifact(
    #     name="iwslt_en-vi_tokenizer_32k", type="tokenizer"
    # )
    # tokenizer_artifact.add_file(local_path=str(config.TOKENIZER_PATH))
    # run.log_artifact(tokenizer_artifact, aliases=["latest"])

    # run.finish()
