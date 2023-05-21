import config
from torch import nn


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is ``(config.NC) x 64 x 64``
            nn.Conv2d(config.NC, config.NDF, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(config.NDF) x 32 x 32``
            nn.Conv2d(config.NDF, config.NDF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.NDF * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(config.NDF*2) x 16 x 16``
            nn.Conv2d(config.NDF * 2, config.NDF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.NDF * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(config.NDF*4) x 8 x 8``
            nn.Conv2d(config.NDF * 4, config.NDF * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.NDF * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(config.NDF*8) x 4 x 4``
            nn.Conv2d(config.NDF * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)
