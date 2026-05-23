import torch
import torch.nn as nn
import torch.optim as optim

X = torch.tensor([[1.], [2.], [3.], [4.], [5.],
                   [6.], [7.], [8.], [9.], [10.]])

y = torch.tensor([[0.], [0.], [0.], [0.], [0.],
                   [1.], [1.], [1.], [1.], [1.]])

class DeepClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Dropout(0, 2),

            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0, 2),

            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Dropout(0, 2),

            nn.Linear(4, 1),
            nn.Sigmoid()

        )
    def forward(self, x):
        return self.network(x)

model = DeepClassifier()

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)

print("Start training with the new data...")
for epoch in range(300):
    predictions = model(X)
    loss = criterion(predictions, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 60 == 0:
        print(f"Epoch [{epoch + 1}/300] - Mistake (Loss): {loss.item():.4f}")

print("\nTraining beendet!\n")

model.eval()
with torch.no_grad():
    final_output = model(X)
    predictions_rounded = torch.round(final_output)

print("Real labels (y):")
print(y.flatten().tolist())
print("Predicted labels of the models:")
print(predictions_rounded.flatten().tolist())

torch.save(model.state_dict(), 'model.pth')
print("Model saved!")

model2 = DeepClassifier()
model2.load_state_dict(torch.load('model.pth'))
print(("Model loaded!"))