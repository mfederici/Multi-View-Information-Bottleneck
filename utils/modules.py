import torch
import torch.nn as nn
from torch.distributions import Normal, Independent
from torch.nn.functional import softplus


# Encoder architecture
class Encoder(nn.Module):
    def __init__(self, z_dim):
        super(Encoder, self).__init__()

        self.z_dim = z_dim

        # Vanilla MLP
        self.net = nn.Sequential(
            nn.Linear(28 * 28, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, z_dim * 2),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        params = self.net(x)

        mu, sigma = params[:, :self.z_dim], params[:, self.z_dim:]
        sigma = softplus(sigma) + 1e-7  # Make sigma always positive

        return Independent(Normal(loc=mu, scale=sigma), 1)  # Return a factorized Normal distribution


class Decoder(nn.Module):
    def __init__(self, z_dim, scale=0.39894):
        super(Decoder, self).__init__()

        self.z_dim = z_dim
        self.scale = scale

        # Vanilla MLP
        self.net = nn.Sequential(
            nn.Linear(z_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 28 * 28)
        )

    def forward(self, z):
        x = self.net(z)
        return Independent(Normal(loc=x, scale=self.scale), 1)


# Auxiliary network for mutual information estimation
class MIEstimator(nn.Module):
    def __init__(self, size1, size2):
        super(MIEstimator, self).__init__()

        # Vanilla MLP
        self.net = nn.Sequential(
            nn.Linear(size1 + size2, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1),
        )

    # Gradient for JSD mutual information estimation and EB-based estimation
    def forward(self, x1, x2):
        pos = self.net(torch.cat([x1, x2], 1))  # Positive Samples
        neg = self.net(torch.cat([torch.roll(x1, 1, 0), x2], 1))
        return -softplus(-pos).mean() - softplus(neg).mean(), pos.mean() - neg.exp().mean() + 1