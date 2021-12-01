from toy import create_data
from torch.utils.data import Dataset


class ToyDataset(Dataset):
    def __init__(self, heteroskedastic=False):
        self.x, self.y = create_data(1000, 0.5, heteroskedastic)
        super().__init__()

    def __len__(self):
        assert len(self.x) == len(self.y)
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]
