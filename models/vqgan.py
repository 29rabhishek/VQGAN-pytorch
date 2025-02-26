import torch
import torch.nn as nn
from models.layers import Encoder
from models.layers import Decoder
from models.layers import Codebook
from collections import OrderedDict

class VQGAN(nn.Module):
    def __init__(self, args):
        super(VQGAN, self).__init__()
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        # self.codebook = Codebook(args).to(device=args.device)
        self.codebook = Codebook(args)
        self.quant_conv = nn.Conv2d(args.latent_dim, args.latent_dim, 1)
        self.post_quant_conv = nn.Conv2d(args.latent_dim, args.latent_dim, 1)

    def forward(self, imgs):
        encoded_images = self.encoder(imgs)
        quant_conv_encoded_images = self.quant_conv(encoded_images) #[256x16x16]
        codebook_mapping, codebook_indices, q_loss = self.codebook(quant_conv_encoded_images)
        post_quant_conv_mapping = self.post_quant_conv(codebook_mapping)
        decoded_images = self.decoder(post_quant_conv_mapping)

        return decoded_images, codebook_indices, q_loss

    def encode(self, imgs):
        encoded_images = self.encoder(imgs)
        quant_conv_encoded_images = self.quant_conv(encoded_images)
        codebook_mapping, codebook_indices, q_loss = self.codebook(quant_conv_encoded_images)
        return codebook_mapping, codebook_indices, q_loss

    def decode(self, z):
        post_quant_conv_mapping = self.post_quant_conv(z)
        decoded_images = self.decoder(post_quant_conv_mapping)
        return decoded_images

    def calculate_lambda(self, perceptual_loss, gan_loss):
        last_layer = self.decoder.model[-1]
        last_layer_weight = last_layer.weight
        perceptual_loss_grads = torch.autograd.grad(perceptual_loss, last_layer_weight, retain_graph=True)[0]
        gan_loss_grads = torch.autograd.grad(gan_loss, last_layer_weight, retain_graph=True)[0]

        λ = torch.norm(perceptual_loss_grads) / (torch.norm(gan_loss_grads) + 1e-4)
        λ = torch.clamp(λ, 0, 1e4).detach()
        return 0.8 * λ

    @staticmethod
    def adopt_weight(disc_factor, i, threshold, value=0.):
        if i < threshold:
            disc_factor = value
        return disc_factor

    # def load_checkpoint(self, path):
    #     self.load_state_dict(torch.load(path))



    def load_checkpoint(self, path):
        # Load the state_dict from the checkpoint
        state_dict = torch.load(path, map_location="cpu")

        # Check if the model was saved with DataParallel (module. prefix)
        new_state_dict = OrderedDict()
        for k, v in state_dict["vqgan_state_dict"].items():
            # Remove 'module.' prefix if it exists
            new_key = k.replace("module.", "") if k.startswith("module.") else k
            new_state_dict[new_key] = v

        # Load the cleaned state_dict into the model
        self.load_state_dict(new_state_dict)

        print(f"Checkpoint loaded successfully from {path}.")










