
from neuralnetwork import NeuralNetwork
import torch

model = NeuralNetwork()

import pandas as pd

### load data
df = pd.read_csv("./data/output_csv_cleaned.csv")

model.load_state_dict(torch.load("./parameters/model.pth"))

x_test = torch.tensor(df.iloc[:,1:].values).float()
import numpy as np
### all possible values of x
x = torch.zeros(2**6,6)
for i in range(2**6):
    x[i] = torch.tensor([int(j) for j in list(np.binary_repr(i,width=6))])

### predict1
print(x.shape)
logits = model.forward(x)
### normalize
logits = logits/torch.sum(logits,1)[:,None]
print(x[0],logits[0])
## save obs probabilities
np.save("parameters/obs_probabilities.npy",logits.detach().numpy())