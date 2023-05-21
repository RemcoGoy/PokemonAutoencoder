import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from config import EPOCHS

from autoencoder.model import AE
from autoencoder.utils import get_data_loader

#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = AE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

data_loader = get_data_loader()

outputs = []
losses = []
for epoch in range(EPOCHS):
    for image, _ in data_loader:
        image = image.view(-1, 64 * 64).to(device)

        # Output of Autoencoder
        reconstructed = model(image)

        # Calculating the loss function
        loss = criterion(reconstructed, image)

        # The gradients are set to zero,
        # the gradient is computed and stored.
        # .step() performs parameter update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Storing the losses in a list for plotting
        losses.append(loss)
    outputs.append((EPOCHS, image, reconstructed))

# Defining the Plot Style
plt.style.use("fivethirtyeight")
plt.xlabel("Iterations")
plt.ylabel("Loss")

# Convert losses list to a tensor
losses_tensor = torch.tensor(losses, device=device)

# Plotting the last 100 values
plt.plot(losses_tensor[-100:].cpu().detach().numpy())

plt.show()
