### conditional gan for binary data

import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, input_dim=10, output_dim=6, hidden_dim=32,n_classes=5):
        super(Generator, self).__init__()
        # input layer
        self.fc1 = nn.Linear(input_dim+1, hidden_dim)
        # hidden layers
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        # output layer
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2) # Leaky ReLU
        x = F.leaky_relu(self.fc2(x), 0.2) # Leaky ReLU
        x = F.leaky_relu(self.fc3(x), 0.2) # Leaky ReLU
        return torch.sigmoid(self.fc4(x)) # Sigmoid

class Discriminator(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=32,n_classes=6):
        super(Discriminator, self).__init__()
        # input layer
        self.fc1 = nn.Linear(input_dim+1, hidden_dim)
        # hidden layers
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        # output layer
        self.fc4 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2) # Leaky ReLU
        x = F.leaky_relu(self.fc2(x), 0.2) # Leaky ReLU
        x = F.leaky_relu(self.fc3(x), 0.2) # Leaky ReLU
        return torch.sigmoid(self.fc4(x)) # Sigmoid

class ConditionalGAN(nn.Module):
    def __init__(self):
        super(ConditionalGAN, self).__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()
    def train(self, data, epochs, batch_size, criterion):
        optimizer_gen = torch.optim.Adam(self.parameters(), lr=0.00002)
        optimizer_disc = torch.optim.Adam(self.parameters(), lr=0.0002)

        for epoch in range(epochs):
            for i in range(0, len(data), batch_size):
                x = data[i:i+batch_size]
                y = torch.rand(batch_size, 10)
                y = torch.cat([y, data[i:i+batch_size,0].reshape(-1,1)], axis=1)
                fake_data = torch.cat([data[i:i+batch_size,0].reshape(-1,1),self.generator(y)],axis=1)
                real_data = data[i:i+batch_size]
                real_output = self.discriminator(real_data)
                fake_output = self.discriminator(fake_data)
                real_loss = criterion(real_output, torch.ones_like(real_output))
                fake_loss = criterion(fake_output, torch.zeros_like(fake_output))                
                d_loss = real_loss + fake_loss
                optimizer_disc.zero_grad()
                d_loss.backward()
                optimizer_disc.step()

                y = torch.rand(batch_size, 10)
                y = torch.cat([y, data[i:i+batch_size,0].reshape(-1,1)], axis=1)
                fake_data = self.generator(y)
                true_data = data[i:i+batch_size,1:]
                g_loss = criterion(fake_data,true_data)
                optimizer_gen.zero_grad()
                g_loss.backward()
               
                if epoch % 100 == 0:
                    print(f"Epoch [{epoch}/{epochs}], d_loss: {d_loss.item():.4f}")
                    print(self.generator(torch.tensor(torch.concat([torch.rand(10),torch.tensor([0])]).float())))
                    print(self.generator(torch.tensor(torch.concat([torch.rand(10),torch.tensor([5])]).float())))

import pandas as pd
if __name__ == "__main__":
    data = pd.read_csv("./data/output_csv_cleaned.csv").values
    data = torch.tensor(data).float()

    cgan = ConditionalGAN()
    if 0:
            
        criterion = nn.BCELoss()
        cgan.train(data, epochs=10000, batch_size=20, criterion=criterion)
        ## save model
        torch.save(cgan.state_dict(), "./parameters/model_cgan.pth")
    else:
        cgan.load_state_dict(torch.load("./parameters/model_cgan.pth"))
    print(data[data[:,0]==0,:])
    for i in range(10):
        print(cgan.generator(torch.tensor(torch.concat([torch.rand(10),torch.tensor([0])]).float())))


    
