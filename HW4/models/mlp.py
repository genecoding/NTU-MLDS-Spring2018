from torch import nn
from torch.distributions import Categorical


class MLP(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, output_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        probs = self.model(x)
        dist = Categorical(probs)
        return dist