# prerequisites
from ast import arg
from tokenize import Double
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from scipy.integrate import odeint



#%% Hyperparameters

bs = 5
n_data = 20
timesteps = 500
slope = 0.01
drop = 0.2
criterion = nn.BCELoss() 
lr =  1e-4
np.random.seed(2022)
#n_epochs = args.n_epochs
#z_dim = args.z_dim
n_epochs = 1
z_dim = 50


def pend(x, t, m, k):
    x1,x2 = x
    dxdt = [x2, -m*x2 - k*np.sin(x1)]
    return dxdt


t = np.linspace(0, 10, timesteps)
   
x = np.zeros((n_data,timesteps))

x0 = [1,0.5]

for i in range(n_data):
    #x0 = [np.random.uniform(0,np.pi),np.random.uniform(0,1)]
    m = np.random.uniform(0.1,2)
    k = np.random.uniform(3,10)
    sol = odeint(pend, x0, t, args=(m, k))
    x[i,:] = sol[:,0]
print('Data generation complete')


   
#%% Ready data

train_set = torch.from_numpy(x)

train_x_u = torch.tensor(x, requires_grad=True).float().to(self.device)
self.train_y_u = torch.tensor(self.y_u, requires_grad=True).float().to(self.device)

#train_set = train_set.float()

#train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=bs, shuffle=True)

#print(train_set.shape)

#%% Define Network




class Generator(nn.Module):
    def __init__(self, g_input_dim, g_output_dim):
        super(Generator, self).__init__()       
        self.fc1 = nn.Linear(g_input_dim, g_input_dim*2)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features*2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features*2)
        self.fc4 = nn.Linear(self.fc3.out_features, self.fc3.out_features*2)
        self.fc5 = nn.Linear(self.fc4.out_features, g_output_dim)
    
    # forward method
    def forward(self, x): 
        x = F.tanh(self.fc1(x)) # leaky relu, with slope angle 
        x = F.tanh(self.fc2(x)) 
        x = F.tanh(self.fc3(x))
        x = F.tanh(self.fc4(x))
        #return torch.tanh(self.fc4(x))
        return self.fc5(x) 
    
class Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, d_input_dim *4)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
        self.fc4 = nn.Linear(self.fc3.out_features, self.fc3.out_features//2)
        self.fc5 = nn.Linear(self.fc4.out_features, 1)  # output dim = 1 for binary classification 
    
    # forward method
    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.dropout(x, drop)
        x = F.tanh(self.fc2(x))
        x = F.dropout(x, drop)
        x = F.tanh(self.fc3(x))
        x = F.dropout(x, drop)
        x = F.tanh(self.fc4(x))
        x = F.dropout(x, drop)
        return torch.sigmoid(self.fc5(x))  # sigmoid for probaility 
        
    
class Q_net(nn.Module):
    def __init__(self, q_input_dim, q_output_dim):
        super(Q_net, self).__init__()       
        self.fc1 = nn.Linear(q_input_dim, q_input_dim*2)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features*2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features*2)
        self.fc4 = nn.Linear(self.fc3.out_features, self.fc3.out_features*2)
        self.fc5 = nn.Linear(self.fc4.out_features, q_output_dim)
        
    def forward(self,x):
        x = F.tanh(self.fc1(x)) # leaky relu, with slope angle 
        x = F.tanh(self.fc2(x)) 
        x = F.tanh(self.fc3(x))
        x = F.tanh(self.fc4(x))
        #return torch.tanh(self.fc4(x))
        return self.fc5(x) 
        


  
# build network
G = Generator(g_input_dim = z_dim, g_output_dim = timesteps).to(device)
D = Discriminator(timesteps).to(device)
Q = Q_net(q_input_dim=z_dim,q_output_dim=timesteps).to(device)

# set optimizer 
G_optimizer = optim.Adam(G.parameters(), lr=lr betas = (0.9, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas = (0.9, 0.999))
Q_optimizer = optim.Adam(Q.parameters(), lr=lr, betas = (0.9, 0.999))


#%% Define training loops 
# 
def G_train(x):
    #=======================Train the generator=======================#
    G.zero_grad()