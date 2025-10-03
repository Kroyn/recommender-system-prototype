import torch
import torch.nn as nn
import numpy as np
import time


# Генерація даних
np.random.seed(42)
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x) + np.random.normal(0, 0.1, 100)
y = (y - y.min()) / (y.max() - y.min())  # Нормалізація до [0,1]
x = torch.FloatTensor(x).reshape(-1, 1)
y = torch.FloatTensor(y).reshape(-1, 1)


# Модель одношарової нейронної мережі
class SingleLayerNN(nn.Module):
    def __init__(self):
        super(SingleLayerNN, self).__init__()
        self.linear = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x


# Модель багатошарової нейронної мережі (2 шари: 3-1, сигмоїда)
class MultiLayerNN(nn.Module):
    def __init__(self):
        super(MultiLayerNN, self).__init__()
        self.layer1 = nn.Linear(1, 3)
        self.layer2 = nn.Linear(3, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.layer1(x))
        x = self.sigmoid(self.layer2(x))
        return x


# Функція для тренування моделі
def train_model(model, lr=0.01, epochs=1000):
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    start_time = time.time()
    for epoch in range(epochs):
        outputs = model(x)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 200 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    end_time = time.time()
    return loss.item(), end_time - start_time


# Тренування одношарової мережі
single_model = SingleLayerNN()
single_loss, single_time = train_model(single_model)
print(f'Single Layer NN - Final Loss: {single_loss:.4f}, Time: {single_time:.2f}s')


# Тренування багатошарової мережі
multi_model = MultiLayerNN()
multi_loss, multi_time = train_model(multi_model)
print(f'Multi Layer NN - Final Loss: {multi_loss:.4f}, Time: {multi_time:.2f}s')
