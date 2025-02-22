import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import utils as vutils
from utils import LPIPS, load_data, weights_init
from models import VQGAN, Discriminator
from torch import nn
import wandb

class TrainVQGAN:
    def __init__(self, args):
        # Initialize wandb
        wandb.init(project="VQGAN-Training", config=args)
        self.args = args

        self.vqgan = VQGAN(args)
        self.discriminator = Discriminator(args)
        self.perceptual_loss = LPIPS()
        self.save_at_idx = args.save_at_idx

        if args.device == "cuda" and torch.cuda.device_count() > 1:
            self.vqgan = nn.DataParallel(self.vqgan)
            self.discriminator = nn.DataParallel(self.discriminator)
            self.perceptual_loss = nn.DataParallel(self.perceptual_loss)

        self.vqgan = self.vqgan.to(device=args.device)
        self.discriminator = self.discriminator.to(device=args.device)

        # Ensure weights initialization is applied properly
        vqgan_model = self.vqgan.module if isinstance(self.vqgan, nn.DataParallel) else self.vqgan
        disc_model = self.discriminator.module if isinstance(self.discriminator, nn.DataParallel) else self.discriminator
        vqgan_model.apply(weights_init)
        disc_model.apply(weights_init)

        self.perceptual_loss = self.perceptual_loss.eval().to(device=args.device)

        self.opt_vq, self.opt_disc = self.configure_optimizers(args)

        self.prepare_training()
        if args.resume:
            self.load_checkpoint()

        self.train(args)

    def configure_optimizers(self, args):
        lr = args.learning_rate
        vqgan_model = self.vqgan.module if isinstance(self.vqgan, torch.nn.DataParallel) else self.vqgan

        opt_vq = torch.optim.Adam(
            list(vqgan_model.encoder.parameters()) +
            list(vqgan_model.decoder.parameters()) +
            list(vqgan_model.codebook.parameters()) +
            list(vqgan_model.quant_conv.parameters()) +
            list(vqgan_model.post_quant_conv.parameters()),
            lr=lr, eps=1e-08, betas=(args.beta1, args.beta2)
        )

        disc_model = self.discriminator.module if isinstance(self.discriminator, torch.nn.DataParallel) else self.discriminator
        opt_disc = torch.optim.Adam(
            disc_model.parameters(),
            lr=lr, eps=1e-08, betas=(args.beta1, args.beta2)
        )

        return opt_vq, opt_disc

    @staticmethod
    def prepare_training():
        os.makedirs("results", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)

    def save_checkpoint(self, epoch, step):
        checkpoint = {
            'vqgan_state_dict': self.vqgan.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'opt_vq_state_dict': self.opt_vq.state_dict(),
            'opt_disc_state_dict': self.opt_disc.state_dict(),
            'epoch': epoch,
            'step': step
        }
        torch.save(checkpoint, os.path.join("checkpoints", "vqgan_checkpoint.pth"))

    def load_checkpoint(self):
        checkpoint_path = os.path.join("checkpoints", "vqgan_checkpoint.pth")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.vqgan.load_state_dict(checkpoint['vqgan_state_dict'])
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            self.opt_vq.load_state_dict(checkpoint['opt_vq_state_dict'])
            self.opt_disc.load_state_dict(checkpoint['opt_disc_state_dict'])
            start_epoch = checkpoint['epoch']
            start_step = checkpoint['step']
            print(f"Resuming training from epoch {start_epoch}, step {start_step}.")
        else:
            print("No checkpoint found. Starting from scratch.")

    def train(self, args):
        train_dataset = load_data(args)
        steps_per_epoch = len(train_dataset)
        for epoch in range(args.epochs):
            with tqdm(range(len(train_dataset))) as pbar:
                for i, imgs in zip(pbar, train_dataset):
                    imgs = imgs.to(device=args.device)
                    decoded_images, _, q_loss = self.vqgan(imgs)

                    disc_real = self.discriminator(imgs)
                    disc_fake = self.discriminator(decoded_images)

                    disc_factor = self.vqgan.module.adopt_weight(args.disc_factor, epoch * steps_per_epoch + i, threshold=args.disc_start) if isinstance(self.vqgan, nn.DataParallel) else self.vqgan.adopt_weight(args.disc_factor, epoch * steps_per_epoch + i, threshold=args.disc_start)

                    perceptual_loss = self.perceptual_loss(imgs, decoded_images)
                    rec_loss = torch.abs(imgs - decoded_images)
                    perceptual_rec_loss = args.perceptual_loss_factor * perceptual_loss + args.rec_loss_factor * rec_loss
                    perceptual_rec_loss = perceptual_rec_loss.mean()
                    g_loss = -torch.mean(disc_fake)

                    λ = self.vqgan.module.calculate_lambda(perceptual_rec_loss, g_loss) if isinstance(self.vqgan, nn.DataParallel) else self.vqgan.calculate_lambda(perceptual_rec_loss, g_loss)
                    vq_loss = perceptual_rec_loss + q_loss + disc_factor * λ * g_loss

                    d_loss_real = torch.mean(F.relu(1. - disc_real))
                    d_loss_fake = torch.mean(F.relu(1. + disc_fake))
                    gan_loss = disc_factor * 0.5 * (d_loss_real + d_loss_fake)

                    self.opt_vq.zero_grad()
                    vq_loss.mean().backward(retain_graph=True)

                    self.opt_disc.zero_grad()
                    gan_loss.mean().backward()

                    self.opt_vq.step()
                    self.opt_disc.step()

                    # Log metrics to wandb
                    wandb.log({
                        "vq_loss": vq_loss.mean().item(),
                        "gan_loss": gan_loss.mean().item(),
                        "epoch": epoch,
                        "step": i,
                        "learning_rate_vq": self.opt_vq.param_groups[0]['lr'],  # Log VQGAN learning rate
                        "learning_rate_disc": self.opt_disc.param_groups[0]['lr']  # Log Discriminator learning rate
                    })

                    if i % self.save_at_idx == 0 and torch.cuda.current_device() == 0:
                        with torch.no_grad():
                            real_fake_images = torch.cat((imgs[:4], decoded_images.add(1).mul(0.5)[:4]))
                            vutils.save_image(real_fake_images, os.path.join("results", f"{epoch}_{i}.jpg"), nrow=4)
                            wandb.log({"reconstructed_images": [wandb.Image(real_fake_images)]})

            # Save checkpoint after each epoch
            self.save_checkpoint(epoch, i)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--image-size', type=int, default=256, help='Image height and width (default: 256)')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors (default: 256)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images (default: 3)')
    parser.add_argument('--dataset-path', type=str, default='txt files/5k_imagenet_train.txt', help='Path to data (default: /data)')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=6, help='Input batch size for training (default: 6)')
    parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate (default: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param (default: 0.0)')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param (default: 0.999)')
    parser.add_argument('--disc-start', type=int, default=10000, help='When to start the discriminator (default: 0)')
    parser.add_argument('--disc-factor', type=float, default=1., help='')
    parser.add_argument('--rec-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1., help='Weighting factor for perceptual loss.')
    parser.add_argument('--save-at-idx', type=int, default=30, help='Save images every n steps.')
    parser.add_argument('--resume', action='store_true', help='Resume training from the last checkpoint.')

    args = parser.parse_args()
    print(f"Using device: {args.device}")

    train_vqgan = TrainVQGAN(args)