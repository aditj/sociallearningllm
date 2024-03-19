
from neuralnetwork import NeuralNetwork
import torch

model = NeuralNetwork()

import pandas as pd

### load data
df = pd.read_csv("./data/output_csv_cleaned.csv")

model.load_state_dict(torch.load("./parameters/model.pth"))

### convert to tensor
test_x = torch.tensor(df.iloc[:,1:].values).float()
test_y = torch.tensor(df['state'].values).long()

print(model.forward(test_x))