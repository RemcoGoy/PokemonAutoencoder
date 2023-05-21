from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_data_loader(image_size=64, batch_size=16):
    """Creates training data loader."""
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_dataset = datasets.ImageFolder("./data/pokemon/", transform)
    return DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
