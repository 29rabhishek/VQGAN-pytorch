import os
import argparse
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torchvision import utils as vutils
import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from discriminator import Discriminator
from lpips import LPIPS
from vqgan import VQGAN
from utils import load_data, weights_init
from lightning.pytorch.loggers import WandbLogger

import wandb

class VQGANTrainer(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        wandb_logger.log_hyperparams(args)

        self.vqgan = VQGAN(args)
        self.discriminator = Discriminator(args)
        self.discriminator.apply(weights_init)
        self.perceptual_loss = LPIPS().eval()
        self.automatic_optimization = False

    def forward(self, imgs):
        return self.vqgan(imgs)

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        
        # Optimizers
        opt_vq = torch.optim.Adam(
            list(self.vqgan.encoder.parameters()) +
            list(self.vqgan.decoder.parameters()) +
            list(self.vqgan.codebook.parameters()) +
            list(self.vqgan.quant_conv.parameters()) +
            list(self.vqgan.post_quant_conv.parameters()),
            lr=lr, eps=1e-08, betas=(self.hparams.beta1, self.hparams.beta2)
        )
        opt_disc = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=lr, eps=1e-08, betas=(self.hparams.beta1, self.hparams.beta2)
        )

        # Schedulers
        sched_vq = StepLR(opt_vq, step_size=30, gamma=0.1)
        sched_disc = StepLR(opt_disc, step_size=30, gamma=0.1)

        return [opt_vq, opt_disc], [sched_vq, sched_disc]

    def training_step(self, batch, batch_idx):
        torch.autograd.set_detect_anomaly(True)
        opt_vq, opt_disc = self.optimizers()
        sched_vq, sched_disc = self.lr_schedulers()
        imgs = batch
        decoded_images, _, q_loss = self.vqgan(imgs)

        # Discriminator Loss
        disc_real = self.discriminator(imgs)
        disc_fake = self.discriminator(decoded_images)

        #have to change this to self.current_epoch*self.trainer.datamodule.train_dataloader()+batch_idx
        disc_factor = self.vqgan.adopt_weight(
            self.hparams.disc_factor, self.global_step, threshold=self.hparams.disc_start
        )

        perceptual_loss = self.perceptual_loss(imgs, decoded_images)
        rec_loss = torch.abs(imgs - decoded_images)
        perceptual_rec_loss = (
            self.hparams.perceptual_loss_factor * perceptual_loss + 
            self.hparams.rec_loss_factor * rec_loss
        ).mean()

        g_loss = -torch.mean(disc_fake)
        
        lambda_g = self.vqgan.calculate_lambda(perceptual_rec_loss, g_loss)
        # VQ Loss
        vq_loss = perceptual_rec_loss + q_loss + disc_factor * lambda_g * g_loss

        # Discriminator Loss
        d_loss_real = torch.mean(F.relu(1. - disc_real))
        d_loss_fake = torch.mean(F.relu(1. + disc_fake))
        gan_loss = disc_factor * 0.5 * (d_loss_real + d_loss_fake)
        

        opt_vq.zero_grad()
        self.manual_backward(vq_loss, retain_graph=True)

        opt_disc.zero_grad()
        self.manual_backward(gan_loss)
       
        opt_vq.step()
        opt_disc.step()

        if self.trainer.is_last_batch:
            sched_vq.step()
            sched_disc.step()

        self.log("train/vq_loss", vq_loss, on_step=True, on_epoch=True)
        self.log("train/gan_loss", gan_loss, on_step=True, on_epoch=True)

    def on_train_epoch_end(self):
        vq_loss = self.trainer.callback_metrics["train/vq_loss"]
        gan_loss = self.trainer.callback_metrics["train/gan_loss"]
        self.log("epoch/vq_loss", vq_loss, on_epoch=True)
        self.log("epoch/gan_loss", gan_loss, on_epoch=True)
        # Save sample images
        with torch.no_grad():
            imgs = next(iter(self.trainer.datamodule.train_dataloader()))[:8].to(self.hparams.device)
            decoded_images, _, _ = self.vqgan(imgs)
            real_fake_images = torch.cat((imgs[:4], decoded_images.add(1).mul(0.5)[:4]))
            vutils.save_image(real_fake_images, os.path.join("results", f"epoch_{self.current_epoch}.jpg"), nrow=4)
            self.logger.experiment.log({
                "sample_images": [wandb.Image(img) for img in real_fake_images]
            })


class VQGANDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def train_dataloader(self):
        return load_data(self.args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--image-size', type=int, default=256, help='Image height and width (default: 256)')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors (default: 256)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images (default: 3)')
    parser.add_argument('--dataset-path', type=str, default='vqgan_eeg_imagenet_5k', help='Path to data (default: /data)')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=6, help='Input batch size for training (default: 6)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate (default: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param (default: 0.0)')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param (default: 0.999)')
    parser.add_argument('--disc-start', type=int, default=10000, help='When to start the discriminator (default: 0)')
    parser.add_argument('--disc-factor', type=float, default=1., help='')
    parser.add_argument('--rec-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1., help='Weighting factor for perceptual loss.')

    args = parser.parse_args()
    wandb_logger = WandbLogger(project="VQGAN", name="vqgan_training", log_model="all")

    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints_vqgan", exist_ok=True)
    # train_dataloader = load_data(args)
    dm = VQGANDataModule(args)
    model = VQGANTrainer(args)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        devices='auto',
        accelerator='auto',
        precision= '32',
        strategy='ddp_spawn',
        default_root_dir="checkpoints_vqgan",
        callbacks=[TQDMProgressBar(refresh_rate=10)],
        logger=wandb_logger
    )

    trainer.fit(model, dm)