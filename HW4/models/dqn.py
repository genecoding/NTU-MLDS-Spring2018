import torch
from torch import nn


class DQN(nn.Module):
    def __init__(self, input_shape, hid_dim, output_dim):
        super().__init__()

        self.input_shape = input_shape

        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, output_dim),
        )

    def feature_size(self):
        return self.cnn(torch.zeros(1, *self.input_shape)).reshape(1, -1).shape[1]

    def forward(self, x):
        x = self.cnn(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x