import csv
from datetime import datetime
import os

class CSVLogger:
    def __init__(self, filename="training_log.csv"):
        self.filename = filename
        self._create_file_if_not_exists()

    def _create_file_if_not_exists(self):
        """Initialize CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.filename):
            with open(self.filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "epoch",
                    "batch",
                    "generator_loss",
                    "critic_loss",
                    "gradient_penalty",
                    "gradient_norm",
                    "generator_lr",
                    "critic_lr"
                ])

    def log(self, epoch, batch, metrics):
        """Append metrics to the CSV file."""
        with open(self.filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                epoch,
                batch,
                metrics.get("generator_loss"),
                metrics.get("critic_loss"),
                metrics.get("gradient_penalty"),
                metrics.get("gradient_norm"),
                metrics.get("generator_lr"),
                metrics.get("critic_lr")
            ])

# Integration:
"""
def train(self):
    logger = CSVLogger("wgan_gp_logs.csv")  # Initialize logger
    
    for epoch in range(self.num_epochs):
        for batch_idx, real_images in enumerate(self.dataloader):
            # ... (existing training code)
            
            # Log metrics after each batch
            logger.log(
                epoch=epoch,
                batch=batch_idx,
                metrics={
                    "generator_loss": G_loss.item(),
                    "critic_loss": C_loss.item(),
                    "gradient_penalty": gradient_penalty.item(),
                    "gradient_norm": grad_norm,
                    "generator_lr": self.optimizerG.param_groups[0]['lr'],
                    "critic_lr": self.optimizerC.param_groups[0]['lr']
                }
            )
"""
# Plotting:
"""
import pandas as pd

df = pd.read_csv("wgan_gp_logs.csv")
print(df.describe())  # Quick stats
df.plot(y=["generator_loss", "critic_loss"])  # Plot trends
"""
