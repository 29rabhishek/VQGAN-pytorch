import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torchvision import utils as vutils
from transformer import VQGANTransformer
from utils import load_transformer_data, plot_images
import wandb

class TrainTransformer:
    def __init__(self, args):
        wandb.init(project="VQGAN-Training", name="transformer-training-experiment-25th-Feb")

        # Set device and initialize model
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        model = VQGANTransformer(args)

        # Wrap model in DataParallel if multiple GPUs are available
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for training.")
            self.model = nn.DataParallel(model).to(self.device)
        else:
            self.model = model.to(self.device)

        # Optimizer, AMP scaler, and scheduler
        self.optim = self.configure_optimizers(args)
        self.scaler = GradScaler()
        self.scheduler = self.configure_scheduler(args)

        # Start training
        self.train(args)

    def configure_optimizers(self, args):
        """Configures optimizer with weight decay settings."""
        decay, no_decay = set(), set()
        whitelist_weight_modules = (nn.Linear,)
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)

        for mn, m in self.model.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn

                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        no_decay.add("pos_emb")
        param_dict = {pn: p for pn, p in self.model.named_parameters()}

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=args.learning_rate, betas=(args.beta1, args.beta2))
        return optimizer

    def configure_scheduler(self, args):
        """Configures the learning rate scheduler."""
        scheduler = torch.optim.lr_scheduler.StepLR(
            self.optim, step_size=10, gamma=0.5
        )
        return scheduler

    def train(self, args):
        """Main training loop with AMP, LR scheduler, and logging."""
        train_dataset = load_transformer_data(args)
        self.model.train()

        for epoch in range(args.epochs):
            total_loss = 0.0

            with tqdm(total=len(train_dataset), desc=f"Epoch {epoch+1}/{args.epochs}") as pbar:
                for i, (eeg, _, imgs) in enumerate(train_dataset):
                    self.optim.zero_grad()

                    # Move data to device
                    imgs = imgs.to(self.device)
                    eeg = eeg.to(self.device)

                    # Forward pass with AMP
                    with torch.amp.autocast():
                        logits, targets = self.model(eeg, imgs)
                        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

                    # Backward pass with AMP
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optim)
                    self.scaler.update()

                    # Log batch loss
                    total_loss += loss.item()
                    pbar.set_postfix(Transformer_Loss=np.round(loss.item(), 4))
                    pbar.update(1)
                    wandb.log({
                        "train/transformer_loss": loss.item(),
                        "train/lr": self.optim.param_groups[0]["lr"],
                        "epoch": epoch + 1,
                        "step": i + 1
                    })

            # Scheduler step after each epoch
            self.scheduler.step()

            # Average epoch loss
            if torch.cuda.current_device() == 0:
                avg_loss = total_loss / len(train_dataset)
                print(f"\nEpoch [{epoch+1}/{args.epochs}] - Avg Loss: {avg_loss:.4f}, LR: {self.optim.param_groups[0]['lr']:.6f}")
                wandb.log({"epoch/avg_loss": avg_loss, "epoch/lr": self.optim.param_groups[0]['lr'], "epoch": epoch+1})

                # Log images
                log, sampled_imgs = self.model.module.log_images(imgs[0][None], eeg[0][None])
                image_path = os.path.join("results", f"transformer_epoch_{epoch+1}.jpg")
                vutils.save_image(sampled_imgs, image_path, nrow=4)
                wandb.log({"epoch/reconstructed_images": [wandb.Image(sampled_imgs, caption=f"Epoch {epoch+1}")]})
                plot_images(log)

            # Save checkpoint (remove DataParallel wrapper)
            # Save only if the model is in DataParallel and on GPU 0
            if not isinstance(self.model, nn.DataParallel) or torch.cuda.current_device() == 0:
                model_to_save = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                checkpoint_path = os.path.join("checkpoints",f"transformer_epoch_{epoch+1}.pt")
                torch.save(model_to_save.state_dict(), checkpoint_path)
                print(f"Checkpoint saved at {checkpoint_path}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN Transformer Training")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z.')
    parser.add_argument('--image-size', type=int, default=256, help='Image height and width.')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors.')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar.')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of image channels.')
    parser.add_argument('--dataset-path', type=str, default='./data', help='Path to dataset.')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/vqgan_checkpoint.pth', help='Checkpoint path.')
    parser.add_argument('--device', type=str, default="cuda", help='Device to use for training.')
    parser.add_argument('--batch-size', type=int, default=6, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs.')
    parser.add_argument('--learning-rate', type=float, default=2.25e-5, help='Learning rate.')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta1.')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta2.')
    parser.add_argument('--disc-start', type=int, default=10000, help='When to start discriminator.')
    parser.add_argument('--disc-factor', type=float, default=1.0, help='Discriminator weight.')
    parser.add_argument('--l2-loss-factor', type=float, default=1.0, help='L2 loss weight.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1.0, help='Perceptual loss weight.')
    parser.add_argument('--pkeep', type=float, default=0.5, help='Percentage of latent codes to keep.')
    parser.add_argument('--sos-token', type=int, default=0, help='Start of sentence token.')

    args = parser.parse_args()

    # Train the transformer
    train_transformer = TrainTransformer(args)
