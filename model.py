import torch.nn as nn

class MLPRegressionModel(nn.Module):
    def __init__(self, n_input_channels=1, n_output_channels=1):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(n_input_channels, 256),
            nn.Sigmoid(),
            nn.Linear(256, n_output_channels)
        )

    def forward(self, x):
        return self.f(x)

class MLPMeanAndVarianceRegressionModel(nn.Module):
    def __init__(self, n_input_channels=1, n_output_channels=1):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(n_input_channels, 256),
            nn.Sigmoid(),
            nn.Linear(256, 64)
        )
        self.mu = nn.Linear(64, 1)
        self.var = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        prediction = self.f(x)
        mu = self.mu(prediction)
        var = self.var(prediction)
        return mu, var
    