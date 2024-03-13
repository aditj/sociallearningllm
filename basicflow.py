import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal

class NormalizingFlow(nn.Module):
    def __init__(self, dim):
        super(NormalizingFlow, self).__init__()
        self.dim = dim
        self.layers = nn.ModuleList([PlanarFlow(dim) for _ in range(5)])  # A simple stack of planar flows

    def forward(self, z):
        log_det_jacobian = 0
        for layer in self.layers:
            z, log_det_j = layer(z)
            log_det_jacobian += log_det_j
        return z, log_det_jacobian

class PlanarFlow(nn.Module):
    def __init__(self, dim):
        super(PlanarFlow, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(1, dim).normal_(0, 0.01))
        self.bias = nn.Parameter(torch.Tensor(1).fill_(0))
        self.scale = nn.Parameter(torch.Tensor(1, dim).normal_(0, 0.01))

    def forward(self, z):
        linear = torch.mm(z, self.weight.t()) + self.bias
        activation = torch.tanh(linear)
        z = z + self.scale * activation

        # Compute log determinant of the Jacobian
        psi = (1 - torch.tanh(linear) ** 2) * self.weight
        log_det_jacobian = torch.log(torch.abs(1 + torch.mm(psi, self.scale.t())))
        return z, log_det_jacobian

def train_normalizing_flow():
    dim = 2  # Dimensionality of the normal distribution
    prior = MultivariateNormal(torch.zeros(dim), torch.eye(dim))  # Prior distribution
    flow = NormalizingFlow(dim=dim)
    optimizer = optim.Adam(flow.parameters(), lr=1e-2)

    # Training loop
    for step in range(10000):
        z0 = prior.sample((128,))  # Sample from the prior
        zk, log_det_jacobian = flow(z0)
        log_prior = prior.log_prob(zk)
        loss = -(log_prior + log_det_jacobian).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item()}")

if __name__ == "__main__":
    train_normalizing_flow()