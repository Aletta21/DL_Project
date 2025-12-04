import torch
import torch.nn as nn


class VAE(nn.Module):
    """Variational Autoencoder for gene expression data."""

    def __init__(self, input_dim: int, latent_dim: int = 100, intermediate_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            nn.BatchNorm1d(intermediate_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(intermediate_dim, intermediate_dim // 2),
            nn.BatchNorm1d(intermediate_dim // 2),
            nn.LeakyReLU(0.2),
        )
        self.fc_mu = nn.Linear(intermediate_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(intermediate_dim // 2, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, intermediate_dim // 2),
            nn.BatchNorm1d(intermediate_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(intermediate_dim // 2, intermediate_dim),
            nn.BatchNorm1d(intermediate_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(intermediate_dim, input_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

def vae_loss(recon, x, mu, logvar, beta=1.0):
    """Return total, reconstruction, and KL losses."""
    recon_loss = nn.MSELoss()(recon, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    total = recon_loss + beta * kl_loss
    return total, recon_loss, kl_loss
