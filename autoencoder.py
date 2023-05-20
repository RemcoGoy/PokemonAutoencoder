import torch
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt

#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
