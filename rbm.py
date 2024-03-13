import torch
from torch import nn
from torch.nn import functional as F


class RBM(nn.Module):
    def __init__(self, num_visible, num_hidden, learning_rate=0.01):
        super(RBM, self).__init__()
        self.w = nn.Parameter(torch.randn(num_visible, num_hidden))  # Weights
        self.v_bias = nn.Parameter(torch.zeros(num_visible))  # Visible bias
        self.h_bias = nn.Parameter(torch.zeros(num_hidden))  # Hidden bias
        self.learning_rate = learning_rate

    def visible_to_hidden(self, v):
        # Calculate probability of hidden units being active
        prob_h = torch.sigmoid(torch.matmul(v, self.w) + self.h_bias)
        return prob_h

    def hidden_to_visible(self, h):
        # Calculate probability of visible units being active
        prob_v = torch.sigmoid(torch.matmul(h, self.w.t()) + self.v_bias)
        return prob_v

    def sample_v_given_h(self, h):
        # Sample visible units based on probabilities
        return torch.bernoulli(self.hidden_to_visible(h))

    def sample_h_given_v(self, v):
        # Sample hidden units based on probabilities
        return torch.bernoulli(self.visible_to_hidden(v))

    def gibbs_sampling(self, v, num_iters=1000):
        # Perform k-step Gibbs sampling
        for _ in range(num_iters):
            h = self.sample_h_given_v(v)
            v = self.sample_v_given_h(h)
        return v
    def contrastive_divergence(self, v_data):
        # Positive phase (reconstruction)
        h_pos = self.sample_h_given_v(v_data)
        v_pos = self.sample_v_given_h(h_pos)

        # Negative phase (sampling)
        h_neg = self.sample_h_given_v(v_pos)
        v_neg = self.sample_v_given_h(h_neg)

        # Calculate positive and negative reconstruction errors
        pos_recon_error = torch.mean(torch.abs(v_data - v_pos))
        neg_recon_error = torch.mean(torch.abs(v_data - v_neg))

        # Update weights and biases using contrastive divergence rule
        self.w.data += self.learning_rate * (torch.matmul(v_data.t(), h_pos) - torch.matmul(v_neg.t(), h_neg))
        self.v_bias.data += self.learning_rate * (torch.mean(v_data - v_pos))
        self.h_bias.data += self.learning_rate * (torch.mean(h_pos - h_neg))

    def train(self, data, epochs=10):
        # Train the RBM for a specified number of epochs
        for epoch in range(epochs):
            for v in data:
                self.contrastive_divergence(v.unsqueeze(0))  # Add a batch dimension

def main():
    # Hyperparameters
    num_visible = 2
    num_hidden = 5
    num_epochs = 20

    # Create RBM model
    model = RBM(num_visible, num_hidden)

    # Sample data (replace with your actual training data)
    probs = torch.concatenate([torch.ones(100,1)/2, torch.ones(100,1)/4], dim=1)
    data = torch.bernoulli(probs)  # Example data
    print(data)
    # Train the RBM
    model.train(data, epochs=num_epochs)
    v = torch.bernoulli(torch.ones(1, num_visible))
    num_samples = 200
    samples = torch.zeros(num_samples, num_visible)
    for _ in range(num_samples):
        samples[_] = model.gibbs_sampling(v.clone(), num_iters=1000)  # Run for 1000 iterations
    print(samples.mean(0))   
    # ... (rest of the code for sample generation)

if __name__ == "__main__":
    main()

import numpy as np
def Agent():
    def __init__(self):
        self.X = 2
        self.prior = np.ones(self.X)/self.X
        self.Y_dim = 5
        self.Y_card = 2
        self.Y = self.Y_card**self.Y_dim

        self.likelihood = np.ones((self.X, self.Y))/self.Y
        self.A = 2
        self.cost = np.identity(self.X)
        assert self.cost.shape == (self.X,self.A)

        self.observed_actions = []
        self.hidden = 3
        self.rbm_prior = RBM(self.X,self.hidden)
    
    def update_observed_actions(self,action):
        self.observed_actions.append(action)
    
    def update_rbm(self):
        self.rbm_prior.train(self.observed_actions)

    def update_prior(self):
        self.prior = self.compute_posterior()

    def compute_posterior(self):
        return self.likelihood * self.prior

    def compute_optimal_action(self):
        self.posterior = self.compute_posterior()
        return np.argmin(np.dot(self.posterior, self.cost))