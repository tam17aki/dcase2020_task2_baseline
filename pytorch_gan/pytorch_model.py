"""
Class definition script of Generative Adversarial Nets in PyTorch.

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
from torch.nn.utils import spectral_norm


class Generator(nn.Module):
    "Generator."

    def __init__(self, x_dim, h_dim, z_dim):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(z_dim, h_dim, normalize=False),
            *block(h_dim, h_dim),
            *block(h_dim, h_dim),
            *block(h_dim, h_dim),
            *block(h_dim, h_dim),
            *block(h_dim, h_dim),
            *block(h_dim, h_dim),
            nn.Linear(h_dim, x_dim),
        )

    def forward(self, noise):
        fake = self.model(noise)
        return fake


class Discriminator(nn.Module):
    "Discriminator."

    def __init__(self, x_dim, h_dim, z_dim, leaky_grad=0.2):
        super(Discriminator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            if normalize:
                layers = [spectral_norm(nn.Linear(in_feat, out_feat))]
            else:
                layers = [nn.Linear(in_feat, out_feat)]
            layers.append(nn.LeakyReLU(leaky_grad, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(x_dim, h_dim),
            *block(h_dim, h_dim),
            *block(h_dim, h_dim),
            *block(h_dim, z_dim),
            nn.Linear(z_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, data):
        validity = self.model(data)

        return validity
