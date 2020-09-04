"""
Class definition script of Variational AutoEncoder in PyTorch.

Copyright (C) 2020 by Akira TAMAMORI

This program is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from torch import nn
import torch


class VariationalAutoEncoder(nn.Module):
    """
    Variational AutoEncoder
    """

    def __init__(self, x_dim, h_dim, z_dim, n_hidden):
        super(VariationalAutoEncoder, self).__init__()

        self.n_hidden = n_hidden  # number of hidden layers

        layers = nn.ModuleList([])
        layers += [nn.Linear(x_dim, h_dim)]
        layers += [nn.Linear(h_dim, h_dim) for _ in range(self.n_hidden)]
        layers += [nn.Linear(h_dim, z_dim), nn.Linear(h_dim, z_dim)]
        self.enc_layers = layers

        layers = nn.ModuleList([])
        layers += [nn.Linear(z_dim, h_dim)]
        layers += [nn.Linear(h_dim, h_dim) for _ in range(self.n_hidden)]
        layers += [nn.Linear(h_dim, x_dim)]
        self.dec_layers = layers

        self.enc_bnorms = nn.ModuleList(
            [nn.BatchNorm1d(h_dim) for _ in range(self.n_hidden + 1)])

        self.dec_bnorms = nn.ModuleList(
            [nn.BatchNorm1d(h_dim) for _ in range(self.n_hidden + 1)])

        # self.relu = nn.ReLU()
        self.relu = nn.LeakyReLU()

    def encoder(self, inputs):
        """
        encoder for VAE
        """

        hidden = self.relu(self.enc_bnorms[0](self.enc_layers[0](inputs)))
        for i in range(self.n_hidden):
            hidden = self.relu(
                self.enc_bnorms[i + 1](self.enc_layers[i + 1](hidden)))
        mean = self.enc_layers[-2](hidden)
        logvar = self.enc_layers[-1](hidden)
        return mean, logvar

    @classmethod
    def reparameterization(cls, mean, logvar):
        """
        Sample latent vector from inputs via reparameterization trick
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        latent = mean + eps * std  # reparameterization trick
        return latent

    def decoder(self, latent):
        """
        decoder for VAE
        """
        hidden = self.relu(self.dec_bnorms[0](self.dec_layers[0](latent)))
        for i in range(self.n_hidden):
            hidden = self.relu(
                self.dec_bnorms[i + 1](self.dec_layers[i + 1](hidden)))
        reconst = self.dec_layers[-1](hidden)
        return reconst

    def forward(self, inputs):
        """
        reconstruct inputs through VAE
        """
        mean, logvar = self.encoder(inputs)
        latent = self.reparameterization(mean, logvar)
        reconst = self.decoder(latent)
        return reconst, mean, logvar

    def get_loss(self, criterion, inputs):
        """
        Calculate loss function of VAE.
        """
        recon_x, mean, logvar = self.forward(inputs)
        xent_loss = criterion(recon_x, inputs)
        beta = 1.0
        kl_loss = beta * (-0.5) * torch.sum(1 + logvar -
                                            mean.pow(2) - logvar.exp())

        return xent_loss + kl_loss, xent_loss, kl_loss
