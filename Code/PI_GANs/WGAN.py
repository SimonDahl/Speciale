# prerequisites
from ast import arg
from tkinter import X
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
import random
import argparse
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from scipy.integrate import odeint, solve_ivp

n_sols = 10
bs = 2
time_limit = 5
n_col = 500

n_neurons = 75
lr = 0.001
z_dim = 1
x_dim = 1
y_dim = 1 
criterion = nn.BCELoss() 
criterion_mse = nn.MSELoss()
n_epochs = 1000

gen_epoch = 5
lambda_phy = 1
lambda_q = 0.5
lambda_val = 0.05
timesteps = 200
t = np.linspace(0,time_limit,timesteps)
SoftAdapt_start = 100
clip_value = 0.01

#%%

k = 5
c = 1
m = 2
def sho(t,y):
    k = np.random.uniform(1,10)
    solution = (y[1],(-(c/m)*y[1]-(k/m)*y[0]))
    return solution


train = np.zeros((n_sols,timesteps))

for i in range(n_sols):
    y_init = [np.random.uniform(1,10),np.random.uniform(1,10)]
    solution = solve_ivp(sho, [0,timesteps], y0 = y_init, t_eval = t)
    sol_data = solution.y[0]
    train[i,:] = sol_data

#%%
t_plot = Variable(torch.from_numpy(t).float(), requires_grad=True).to(device)

train_set = torch.from_numpy(train)

train_set = train_set.float()

train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=bs, shuffle=True)



#%%
class Generator(nn.Module):
    def __init__(self, g_input_dim, g_output_dim):
        super(Generator, self).__init__()       
        self.fc1 = nn.Linear(g_input_dim, n_neurons)
        self.fc2 = nn.Linear(n_neurons, n_neurons)
        self.fc3 = nn.Linear(n_neurons, n_neurons)
        self.fc4 = nn.Linear(n_neurons, n_neurons)
        self.fc5 = nn.Linear(n_neurons, g_output_dim)
        
    # forward method
    def forward(self,y):
        y = torch.tanh(self.fc1(y)) # leaky relu, with slope angle 
        y = torch.tanh(self.fc2(y)) 
        y = torch.tanh(self.fc3(y))
        y = torch.tanh(self.fc4(y))
        return self.fc5(y)
    
class Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, n_neurons)
        self.fc2 = nn.Linear(n_neurons,n_neurons)
        self.fc3 = nn.Linear(n_neurons,n_neurons)
        self.fc4 = nn.Linear(n_neurons, 1)  # output dim = 1 for binary classification 
    
    # forward method
    def forward(self, d):
        d = torch.tanh(self.fc1(d))
        d = torch.tanh(self.fc2(d))
        d = torch.tanh(self.fc3(d))
        return self.fc4(d)



#%%
G = Generator(g_input_dim = timesteps, g_output_dim = timesteps).to(device)
D = Discriminator(timesteps).to(device)


# set optimizer 
G_optimizer = optim.Adam(G.parameters(), lr=lr)
D_optimizer = optim.Adam(D.parameters(), lr=lr)

#G_optimizer = optim.RMSprop(G.parameters(), lr=lr)
#D_optimizer = optim.RMSprop(D.parameters(), lr=lr)

#%%
def wasserstein_loss_G(predict):
    loss = -torch.mean(predict)
    return loss

def wasserstein_loss_D(true,predict):
    loss = torch.mean(predict) - torch.mean(true)
    #loss = -torch.mean(true + predict)
    return loss

#%% Define training loops 
# 
def G_train():
    #=======================Train the generator=======================#
    G_optimizer.zero_grad()

    z = Variable(torch.randn(bs, timesteps).to(device))
    
    y_pred= G(z)
    D_output = D(y_pred)
    
    #G_loss = criterion(D_output, y)
    G_loss = wasserstein_loss_G(D_output)
    G_loss.backward()
    G_optimizer.step()
    return G_loss

def D_train(y):
    #=======================Train the discriminator=======================#
    D_optimizer.zero_grad()
    
    # train discriminator on real data
  
    D_real = D(y)
    z = Variable(torch.randn(bs, timesteps).to(device))

    y_pred = G(z)
    D_fake = D(y_pred)
    
    D_loss = wasserstein_loss_D(D_real,D_fake)

    
    D_loss.backward()
    D_optimizer.step()

    for p in D.parameters():
            p.data.clamp_(-clip_value, clip_value)

    return  D_loss

#%%
for epoch in range(1, n_epochs+1):           
    D_losses, G_losses = [], []
    for batch_idx,data in enumerate(train_loader):
     
        D_losses.append(D_train(data))
        G_losses.append(G_train())

    print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
            (epoch), n_epochs, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))

#%%
with torch.no_grad():
    for i in range(3):
        z = Variable(torch.randn(timesteps).to(device))
        G_input = torch.cat((t_plot,z))
        generated = G(z)
        y = generated.cpu().detach().numpy()
        plt.plot(t,y)
    plt.show()
    
    