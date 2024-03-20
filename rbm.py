import torch
from torch import nn
from torch.nn import functional as F
import pandas as pd
import numpy as np
import tqdm 

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
    num_visible = 6
    num_hidden = 4
    num_epochs = 100
    
    
    # Create RBM model

    # Sample data (replace with your actual training data)
    data = pd.read_csv("./data/output_csv_cleaned.csv").values
    n_classes = np.unique(data[:,0]).shape[0]
    for j in tqdm.tqdm(range(n_classes)):
        data_class = data[data[:,0]==j]
        print(data_class.shape)
        data_binary_class = data_class[:,1:]
        data_binary_class = torch.tensor(data_binary_class).float()
        model = RBM(num_visible, num_hidden)

        model.train(data_binary_class, epochs=num_epochs)
        ### save model
        torch.save(model.state_dict(), f"./parameters/model_rbm_{j}.pth")
    
    ### use ohe for non-binary data
    
    # Train the RBM

    # ... (rest of the code for sample generation)

if __name__ == "__main__":
    if 0: 
        main()
    else:
        num_visible = 6
        num_hidden = 4
        num_classes = 6
        num_obs_vars = 6
        obs_probs = np.zeros((num_classes, 2**num_obs_vars))

        for j in range(num_classes):
            model = RBM(num_visible, num_hidden)
            model.load_state_dict(torch.load(f"./parameters/model_rbm_{j}.pth"))
            v = torch.bernoulli(torch.ones(1, num_visible))
            num_samples = 1000
            samples = torch.zeros(num_samples, num_visible)

            for _ in tqdm.tqdm(range(num_samples)):
                samples[_] = model.gibbs_sampling(v.clone(), num_iters=200) 

            # np.save("parameters/samples.npy",samples.detach.numpy())
            for _ in range(num_samples):
                ### convert from binary to decimal
                obs_sample = sum([2**i for i in range(num_obs_vars) if samples[_][i] == 1])
                obs_probs[j,obs_sample] += 1
            obs_probs[j] /= num_samples
            print(obs_probs[j])
            print(samples.mean(0))  
        np.save("parameters/obs_probs.npy",obs_probs) 
