import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.optim import Adam
from torch.utils.data import DataLoader
from model import MLPRegressionModel
from dataset import ToyDataset


class Trainer:
    def __init__(self, quantile, device="cpu"):
        self.quantile = quantile
        self.device = torch.device(device)
        self.model = MLPRegressionModel()
        self.optimizer = Adam(self.model.parameters(), lr=1e-3)

        self.model.to(self.device)

    def train(self, dataset, n_epochs):
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        for i in range(n_epochs):
            epoch_losses = []
            for x, y in dataloader:
                x = x.float().to(self.device).unsqueeze(1)
                y = y.float().to(self.device).unsqueeze(1)

                ypred = self.model(x)
                loss = self.quantile_huber_loss(ypred-y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_losses.append(loss.item())
            print(f"Epoch {i} Loss: {np.mean(epoch_losses)}")

    def quantile_huber_loss(self, errors, k=1.0):
        return torch.where(
            errors < -self.quantile * k, 
            self.quantile * errors.abs(),
            torch.where(
                errors > (1. - self.quantile) * k,
                (1. - self.quantile) * errors.abs(),
                (1. / (2 * k)) * errors ** 2
            )
        ).mean()


def evaluate(trainer, dataset):
    def predict(data):
        x = torch.as_tensor(data.x).float().to(trainer.device).unsqueeze(1)
        with torch.no_grad():
            y = trainer.model(x)
        return y.to(torch.device("cpu")).numpy()

    plt.plot(
        dataset.x, predict(dataset), linewidth=4,
        label=f"quantile={trainer.quantile}"
    )


if __name__ == "__main__":
    trainer1 = Trainer(quantile=0.05)
    trainer2 = Trainer(quantile=0.50)
    trainer3 = Trainer(quantile=0.95)
    
    data = ToyDataset()

    trainer1.train(data, 100)
    trainer2.train(data, 100)
    trainer3.train(data, 100)

    plt.scatter(data.x, data.y, s=1, c="k")
    evaluate(trainer1, data)
    evaluate(trainer2, data)
    evaluate(trainer3, data)

    plt.legend()
    plt.show()