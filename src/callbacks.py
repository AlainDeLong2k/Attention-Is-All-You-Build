class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss):
        # Check improvement
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        # Stop condition
        if self.counter >= self.patience:
            self.should_stop = True
            if self.verbose:
                print(
                    f"[EarlyStopping] No improvement for {self.patience} epochs → stopping."
                )
