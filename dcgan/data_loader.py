import config
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

dataset = dset.ImageFolder(
    root=config.DATA,
    transform=transforms.Compose(
        [
            transforms.Resize(config.IMG_SIZE),
            transforms.CenterCrop(config.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    ),
)

dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.WORKERS
)
