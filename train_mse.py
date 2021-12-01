import ipdb
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.optim import Adam
from torch.utils.data import DataLoader
from model import MLPRegressionModel
from dataset import ToyDataset

class Trainer:
    def __init__(self, device="cpu") -> None:
        self.device = torch.device(device)
        self.model = MLPRegressionModel()
        self.optimizer = Adam(self.model.parameters(), lr=1e-3)

        self.model.to(self.device)

    def train(self, n_epochs):
        dataset = ToyDataset()
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        for i in range(n_epochs):
            epoch_losses = []
            for x, y in dataloader:
                x = x.float().to(self.device).unsqueeze(1)
                y = y.float().to(self.device).unsqueeze(1)

                ypred = self.model(x)
                loss = F.mse_loss(ypred, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_losses.append(loss.item())
            print(f"Epoch {i} Loss: {np.mean(epoch_losses)}")

    def evaluate(self):
        def predict(data):
            x = torch.as_tensor(data.x).float().to(self.device).unsqueeze(1)
            with torch.no_grad():
                y = self.model(x)
            return y.to(torch.device("cpu")).numpy()
        
        dataset = ToyDataset()
        plt.scatter(dataset.x, dataset.y)
        plt.plot(dataset.x, predict(dataset), linewidth=4, color="black")
        plt.show()


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train(200)
    trainer.evaluate()


    
