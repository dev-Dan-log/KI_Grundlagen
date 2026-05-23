import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

transform = transforms.ToTensor()

from torchvision.datasets import MNIST
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

train_data = datasets.MNIST(root='data', train = True, download=True, transform=transform)
test_data = datasets.MNIST(root='data', train = False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

class MNISTNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 10)

        )
    def forward(self, x):
        return self.network(x)

model = MNISTNetwork().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"Netz bereit auf: {device}")

for epoch in range(5):
    model.train()
    for batch_X,  batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_X)
            predicted = torch.argmax(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted.cpu() == batch_y.cpu()).sum().item()

    print(f"Epoch [{epoch+1}/5] - Genauigkeit: {100 * correct / total:.2f}%")
