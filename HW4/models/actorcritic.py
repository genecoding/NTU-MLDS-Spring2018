from torch import nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, output_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, 1),
        )

    def forward(self, x):
        value = self.critic(x)
        logits = self.actor(x)
        dist = Categorical(logits=logits)
        return dist, value