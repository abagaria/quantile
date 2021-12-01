import numpy as np


def create_data(num_points, noise_std, heteroskedastic=False):
    x = np.linspace(0, 2*np.pi, num=num_points)
    noise = np.random.normal(0, noise_std, size=x.shape)
    if heteroskedastic:
        noise *= (x ** 0.5)
    y = np.sin(x) + noise
    return x, y
