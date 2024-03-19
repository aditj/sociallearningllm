
import torch 
import torch.nn as nn
### Feedforward network for input of 6 binary features and output of 6 classes
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(6, 4),
            nn.ReLU(),
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, 6),
            nn.Softmax(dim=1)
        )

    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)

        return logits
    
    def loss(self, logits, labels):
        return nn.CrossEntropyLoss()(logits, labels)
    
    def accuracy(self, logits, labels):
        return (logits.argmax(1) == labels).float().mean()
    

model = NeuralNetwork()

import pandas as pd
### load data 
df = pd.read_csv("./data/output_csv_cleaned.csv")

### split data into train and test
train = df.sample(frac=0.8, random_state=200)
test = df.drop(train.index)

### convert to tensor
train_x = torch.tensor(train.iloc[:,1:].values).float()
train_y = torch.tensor(train['state'].values).long()
test_x = torch.tensor(test.iloc[:,1:].values).float()
test_y = torch.tensor(test['state'].values).long()

### train model
learning_rate = 1e-4
batch_size = 20
epochs = 100000
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
if True:
    for epoch in range(epochs):
        for i in range(0, len(train_x), batch_size):
            x = train_x[i:i+batch_size]
            y = train_y[i:i+batch_size]
            logits = model(x)
            loss = model.loss(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss: {loss.item()}")
            print(f"Accuracy: {model.accuracy(logits, y)}")

            ### test model
            test_logits = model(test_x)
            test_loss = model.loss(test_logits, test_y)
            print(f"Test Loss: {test_loss.item()}")
            print(f"Test Accuracy: {model.accuracy(test_logits, test_y)}")

        ### save model
        torch.save(model.state_dict(), "./parameters/model.pth")

### load model
model = NeuralNetwork()
model.load_state_dict(torch.load("./parameters/model.pth"))


    
