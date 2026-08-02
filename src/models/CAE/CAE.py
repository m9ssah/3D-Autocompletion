import torch
import torch.nn as nn


class Conv3dAE(nn.Module):
    """
    3D convolutional autoencoder for SDF grids

    Three stride-2 conv layers dowwnsample input to 1/8 of its original size, then a linear layer maps to the latent space.

    kernel_size=4, stride=2, padding=1 is used to ensure that the output size is half of the input size for each conv layer.

    Output layer is linear
    """

    def __init__(self, input_size=48, latent_dim=32):
        super().__init__()
        assert input_size % 8 == 0, "Input size must be divisible by 8"

        self.spactial_dim = input_size // 8
        self.flat_size = 64 * self.spatial_dim**3

        self.encoder_conv = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        self.to_latent = nn.Linear(self.flat_size, latent_dim)

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(16, 1, kernel_size=4, stride=2, padding=1),
        )

    def encode(self, x):
        pass

    def decode(self, z):
        pass

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
