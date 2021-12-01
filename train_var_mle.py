import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.optim import Adam
from torch.utils.data import DataLoader
from model import MLPMeanAndVarianceRegressionModel
from dataset import ToyDataset


class Trainer:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.model = MLPMeanAndVarianceRegressionModel()
        self.optimizer = Adam(self.model.parameters(), lr=1e-3)

        self.model.to(self.device)

    def train(self, n_epochs):
        dataset = ToyDataset(heteroskedastic=True)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        for i in range(n_epochs):
            epoch_losses = []
            for x, y in dataloader:
                x = x.float().to(self.device).unsqueeze(1)
                y = y.float().to(self.device).unsqueeze(1)

                mean_pred, var_pred = self.model(x)
                loss = F.gaussian_nll_loss(input=mean_pred,
                                           target=y,
                                           var=var_pred)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_losses.append(loss.item())
            print(f"Epoch {i} Loss: {np.mean(epoch_losses)}")

    def evaluate(self):
        def predict(data):
            x = torch.as_tensor(data.x).float().to(self.device).unsqueeze(1)
            with torch.no_grad():
                mean, var = self.model(x)
            return mean.to(torch.device("cpu")).numpy(),\
                    var.to(torch.device("cpu")).numpy()
        
        dataset = ToyDataset(heteroskedastic=True)
        mu, sigma_sq = predict(dataset)
        mu = mu.squeeze(); sigma_sq = sigma_sq.squeeze()

        plt.scatter(dataset.x, dataset.y)
        plt.plot(dataset.x, mu, linewidth=4, color="black")
        plt.fill_between(dataset.x, mu-sigma_sq, mu+sigma_sq, alpha=0.3)
        plt.show()


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train(100)
    trainer.evaluate()


    
