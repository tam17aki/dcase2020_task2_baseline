from torch import nn


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.original_dim = 640
        self.latent_dim = 8
        self.input_layer = nn.Linear(640, 128)
        self.interm_layer = nn.Linear(128, 128)
        self.interm2latent_layer = nn.Linear(128, 8)
        self.latent2interm_layer = nn.Linear(8, 128)
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.batchnorm2 = nn.BatchNorm1d(8)
        self.output_layer = nn.Linear(128, 640)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.batchnorm1(self.relu(self.input_layer(x)))
        h = self.batchnorm1(self.relu(self.interm_layer(h)))
        h = self.batchnorm1(self.relu(self.interm_layer(h)))
        h = self.batchnorm1(self.relu(self.interm_layer(h)))
        h = self.batchnorm2(self.relu(self.interm2latent_layer(h)))
        h = self.batchnorm1(self.relu(self.latent2interm_layer(h)))
        h = self.batchnorm1(self.relu(self.interm_layer(h)))
        h = self.batchnorm1(self.relu(self.interm_layer(h)))
        h = self.batchnorm1(self.relu(self.interm_layer(h)))
        x = self.output_layer(h)

        return x
