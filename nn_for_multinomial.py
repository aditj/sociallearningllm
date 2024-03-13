##### Neural Networks to approximate multinomial distribution given observations from them 

import torch
import torch.nn as nn
import torch.optim as optim

### Restricted Boltzmann Machine
class RBM(nn.Module):
    def __init__(self, n_visible, n_hidden, k=1):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible))
        self.v_bias = nn.Parameter(torch.randn(n_visible))
        self.h_bias = nn.Parameter(torch.randn(n_hidden))
        self.k = k
    

    def sample_from_p(self, p):
        return torch.bernoulli(p)

    def v_to_h(self, v):
        p_h = torch.sigmoid(torch.matmul(v, self.W.t()) + self.h_bias)
        return p_h, self.sample_from_p(p_h)

    def h_to_v(self, h):
        p_v = torch.sigmoid(torch.matmul(h, self.W) + self.v_bias)
        return p_v, self.sample_from_p(p_v)

    def forward(self, v):
        h_p, h_s = self.v_to_h(v)
        for _ in range(self.k):
            v_p, v_s = self.h_to_v(h_s)
            h_p, h_s = self.v_to_h(v_s)
        return v, v_s

    def free_energy(self, v):
        vbias_term = v.mv(self.v_bias)
        wx_b = torch.matmul(v, self.W.t()) + self.h_bias
        hidden_term = wx_b.exp().add(1).log().sum(1)
        return (-hidden_term - vbias_term).mean()
    
    def fit(self, v, lr=0.1, epochs=10):
        optimizer = optim.SGD(self.parameters(), lr=lr)
        for epoch in range(epochs):
            for batch in v:
                batch = batch.view(-1, self.W.size(1))
                v, v_s = self.forward(batch)
                loss = self.free_energy(v) - self.free_energy(v_s)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch+1}/{epochs} Loss: {loss.item()}')
        return self
    
    def predict(self, v):
        return self.forward(v)[1]
    
    def reconstruct(self, v):
        return self.forward(v)[0]
    

import numpy as np

sample_data = np.random.binomial(1,0.2, size=(1000,1))
sample_data = torch.tensor(sample_data, dtype=torch.float32)

rbm = RBM(1, 2)
rbm.fit(sample_data, epochs=30)
print(sample_data.mean(0))
print(rbm.predict(sample_data).mean(0))
