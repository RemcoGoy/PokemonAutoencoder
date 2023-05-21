import multiprocessing
import random

from matplotlib import animation

import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils
from data_loader import dataloader
from discriminator import Discriminator
from generator import Generator
from torch import nn, optim
from utils import weights_init


def main():
    # Set reproducability
    manualSeed = 123
    print("Random seed:", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    device = torch.device(
        "cuda:0" if (torch.cuda.is_available() and config.NGPU > 0) else "cpu"
    )
    print("Device:", device)

    # Plot some training images
    # real_batch = next(iter(dataloader))
    # plt.figure(figsize=(8, 8))
    # plt.axis("off")
    # plt.title("Training Images")
    # plt.imshow(
    #     np.transpose(
    #         vutils.make_grid(
    #             real_batch[0].to(device)[:64], padding=2, normalize=True
    #         ).cpu(),
    #         (1, 2, 0),
    #     )
    # )

    netG = Generator(config.NGPU).to(device)

    # Multi GPU
    if (device.type == "cuda") and (config.NGPU > 1):
        netG = torch.nn.DataParallel(netG, list(range(config.NGPU)))

    # Set initial weights
    netG.apply(weights_init)

    # print(netG)

    netD = Discriminator(config.NGPU).to(device)

    # Multi GPU
    if (device.type == "cuda") and (config.NGPU > 1):
        netD = torch.nn.DataParallel(netD, list(range(config.NGPU)))

    #  Set initial weights
    netD.apply(weights_init)

    # print(netD)

    criterion = nn.BCELoss()

    fixed_noise = torch.randn(64, config.NZ, 1, 1, device=device)

    real_label = 1
    fake_label = 0

    optimizerD = optim.Adam(
        netD.parameters(), lr=config.LR, betas=(config.BETA1, 0.999)
    )
    optimizerG = optim.Adam(
        netG.parameters(), lr=config.LR, betas=(config.BETA1, 0.999)
    )

    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")

    for epoch in range(config.EPOCHS):
        for i, data in enumerate(dataloader, 0):
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ## All real batch
            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            ## All fake batch
            noise = torch.randn(b_size, config.NZ, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            # (2) Update G network: maximize log(D(G(z)))
            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print(
                    "[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f"
                    % (
                        epoch,
                        config.EPOCHS,
                        i,
                        len(dataloader),
                        errD.item(),
                        errG.item(),
                        D_x,
                        D_G_z1,
                        D_G_z2,
                    )
                )

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or (
                (epoch == config.EPOCHS - 1) and (i == len(dataloader) - 1)
            ):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(
        fig, ims, interval=1000, repeat_delay=1000, blit=True
    )

    plt.show()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
