import config
from torch import nn


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(config.NZ, config.NGF * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(config.NGF * 8),
            nn.ReLU(True),
            # state size. ``(config.NGF*8) x 4 x 4``
            nn.ConvTranspose2d(config.NGF * 8, config.NGF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.NGF * 4),
            nn.ReLU(True),
            # state size. ``(config.NGF*4) x 8 x 8``
            nn.ConvTranspose2d(config.NGF * 4, config.NGF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.NGF * 2),
            nn.ReLU(True),
            # state size. ``(config.NGF*2) x 16 x 16``
            nn.ConvTranspose2d(config.NGF * 2, config.NGF, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.NGF),
            nn.ReLU(True),
            # state size. ``(config.NGF) x 32 x 32``
            nn.ConvTranspose2d(config.NGF, config.NC, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, input):
        return self.main(input)
