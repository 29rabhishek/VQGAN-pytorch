import torch
import torch.nn as nn
import torch.nn.functional as F
# from models import GPT
from models import GPT2 as GPT ##GPT with cross attentation
from models import VQGAN


class VQGANTransformer(nn.Module):
    def __init__(self, args, eeg_model, vqgan):
        super(VQGANTransformer, self).__init__()

        self.sos_token = args.sos_token

        #add logic to load the model to generate eeg embeddings
        self.eeg_model = eeg_model
        self.eeg_model.register_latent_hook()
        self.vqgan = vqgan

        transformer_config = {
            "vocab_size": args.num_codebook_vectors,
            "block_size": 512,
            "n_layer": 24,
            "n_head": 16,
            "n_embd": 1024,
            "context_dim": 1088 # cross attention query dim
        }
        self.transformer = GPT(**transformer_config)

        self.pkeep = args.pkeep

    @torch.no_grad()
    def encode_to_z(self, x):
        quant_z, indices, _ = self.vqgan.encode(x)
        indices = indices.view(quant_z.shape[0], -1)
        return quant_z, indices

    @torch.no_grad()
    def z_to_image(self, indices, p1=16, p2=16):
        ix_to_vectors = self.vqgan.codebook.embedding(indices).reshape(indices.shape[0], p1, p2, 256)
        ix_to_vectors = ix_to_vectors.permute(0, 3, 1, 2)
        image = self.vqgan.decode(ix_to_vectors)
        return image
    
    @torch.no_grad()
    def get_eeg_embed(self, c):
        _ = self.eeg_model(c)
        return self.eeg_model.activations['latent'][0]

    def forward(self, c, x):# need to add code to call model to generate eeg embeddings
        eeg_latent_embed = self.get_eeg_embed(c) # Extracting Latent EEG embed n

        _, indices = self.encode_to_z(x)#[bs, 256]

        sos_tokens = torch.ones(x.shape[0], 1) * self.sos_token
        sos_tokens = sos_tokens.long().to(indices.device)

        mask = torch.bernoulli(self.pkeep * torch.ones(indices.shape, device=indices.device))
        mask = mask.round().to(dtype=torch.int64)
        random_indices = torch.randint_like(indices, self.transformer.config.vocab_size)
        new_indices = mask * indices + (1 - mask) * random_indices

        new_indices = torch.cat((sos_tokens, new_indices), dim=1)

        target = indices

        logits, _ = self.transformer(new_indices[:, :-1], eeg_latent_embed)#change to include eeg embeddings
        #target shape[bs,256]
        return logits, target

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float("inf")
        return out
    
    def get_device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def sample(self, x, c, eeg, steps, temperature=1.0, top_k=100):
        self.transformer.eval()
        eeg_latent_embed = self.get_eeg_embed(eeg)
        x = torch.cat((c, x), dim=1)
        for k in range(steps):
            logits, _ = self.transformer(x, eeg_latent_embed)
            logits = logits[:, -1, :] / temperature
#n13054560
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)

            probs = F.softmax(logits, dim=-1)

            ix = torch.multinomial(probs, num_samples=1)

            x = torch.cat((x, ix), dim=1)

        x = x[:, c.shape[1]:]
        self.transformer.train()
        return x

    @torch.no_grad()
    def log_images(self, x, eeg):
        log = dict()

        _, indices = self.encode_to_z(x)
        sos_tokens = torch.ones(x.shape[0], 1) * self.sos_token
        sos_tokens = sos_tokens.long().to(self.get_device())

        start_indices = indices[:, :indices.shape[1] // 2]
        sample_indices = self.sample(start_indices, sos_tokens, eeg, steps=indices.shape[1] - start_indices.shape[1])
        half_sample = self.z_to_image(sample_indices)

        start_indices = indices[:, :0]
        sample_indices = self.sample(start_indices, sos_tokens, eeg, steps=indices.shape[1])
        full_sample = self.z_to_image(sample_indices)

        x_rec = self.z_to_image(indices)

        log["input"] = x
        log["rec"] = x_rec
        log["half_sample"] = half_sample
        log["full_sample"] = full_sample

        return log, torch.concat((x, x_rec, half_sample, full_sample))
















