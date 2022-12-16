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
import time 
import argparse
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from scipy.integrate import solve_ivp


n_neurons = 50
lr = 0.001 # learing rate
D_epochs =  1
bs = 5


criterion_MSE = nn.MSELoss() # loss function
criterion_BCE = nn.BCELoss()  
n_epochs = 500
n_col = 1000 # number of collocation points 
lambda_phy = 1 # physics loss weight


timesteps = 500 # number of timestpes for solver
train_size = 50
time_limit = 10 # solver time limit 


m = 2
k = 5
c = 1

t = np.linspace(0,time_limit,timesteps)
    
def sho(t,y):
    solution = (y[1],(-(c/m)*y[1]-(k/m)*y[0])) # damped harmonic oscillator 
    return solution
    
y_init = [3,1] # initial condition
solution = solve_ivp(sho, [0,timesteps], y0 = y_init, t_eval = t) 
sol_data = solution.y[0]
sol_plot = np.array([sol_data])  


x_col = np.linspace(0, time_limit, n_col)
x_col = x_col.reshape(n_col,1)
x_col = Variable(torch.from_numpy(x_col).float(), requires_grad=True).to(device)


""" plt.plot(t,sol_plot[0,:])
plt.show()
 """
 
rand_idx = np.random.choice(timesteps, size=train_size, replace=False, )
 

t_train = np.array(t)[rand_idx.astype(int)]
u_train = np.array(sol_data)[rand_idx.astype(int)]


t_train = t_train.reshape(train_size,1)
t_train = Variable(torch.from_numpy(t_train).float(), requires_grad=True).to(device)

u_train = u_train.reshape(train_size,1)
u_train = Variable(torch.from_numpy(u_train).float(), requires_grad=True).to(device)

t_test = t.reshape(timesteps,1)
t_test = Variable(torch.from_numpy(t_test).float(), requires_grad=True).to(device)

""" plt.scatter(t_train,y_train)
plt.show() """
 

class Generator(nn.Module):
    def __init__(self, g_input_dim, g_output_dim):
        super(Generator, self).__init__()       
        self.fc1 = nn.Linear(g_input_dim, n_neurons)
        self.fc2 = nn.Linear(n_neurons, n_neurons)
        self.fc3 = nn.Linear(n_neurons, n_neurons)
        self.fc4 = nn.Linear(n_neurons, g_output_dim)
        
      
    # forward method
    def forward(self,y):
        y = F.silu(self.fc1(y)) 
        y = F.silu(self.fc2(y)) 
        y = F.silu(self.fc3(y))
        return self.fc4(y) 

class Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, n_neurons)
        self.fc2 = nn.Linear(n_neurons,n_neurons)
        self.fc3 = nn.Linear(n_neurons, 1)  # output dim = 1 for binary classification 
    
    # forward method
    def forward(self, d):
        d = torch.relu(self.fc1(d))
        d = torch.relu(self.fc2(d))
        return self.fc3(d)


#  build networks
G = Generator(g_input_dim =1, g_output_dim = 1).to(device)
D = Discriminator(d_input_dim = 2).to(device)
    
# set optimizer 
G_optimizer = optim.Adam(G.parameters(), lr=lr)
D_optimizer = optim.Adam(D.parameters(), lr=lr)
        
# Physics-Informed residual on the collocation points         
def compute_residuals(x_collocation):
    u = G(x_collocation)           
   
    u_t  = torch.autograd.grad(u, x_collocation, torch.ones_like(u), retain_graph=True,create_graph=True)[0]# computes dy/dx
    u_tt = torch.autograd.grad(u_t,  x_collocation, torch.ones_like(u_t),retain_graph=True ,create_graph=True)[0]# computes d^2y/dx^2
    
    r_ode = m*u_tt+c*u_t + k*u
    return r_ode


def G_train(t_train,u_real,x_collocation):
    
    G.zero_grad()
    
    # MSE loss on training points
    u_fake = G(t_train)
    L_MSE = criterion_MSE(u_real,u_fake)
    
    
    phy_target = torch.zeros_like(x_collocation)
    res = compute_residuals(x_collocation)
    L_phy = criterion_MSE(res,phy_target)
    
    d_input = torch.cat((t_train,u_fake),dim=1)     
    L_adv = torch.mean(D(d_input)) 
           
    G_loss = L_adv + L_MSE + lambda_phy * L_phy 
    
    G_loss.backward()
    G_optimizer.step()
    return G_loss

def D_train(t_train,u_real):
    
    for d_epoch in range(D_epochs):
        D.zero_grad()
        
        u_fake = G(t_train)
        d_input = torch.cat((t_train,u_fake),dim=1)
        u_fake = D(d_input)

        d_input = torch.cat((t_train,u_real),dim=1)
        u_real = D(d_input)
        
        D_loss = torch.mean(u_real)-torch.mean(u_fake)
        
        D_loss.backward()
        D_optimizer.step()
    
    return D_loss

    
for epoch in range(1, n_epochs+1):
    D_losses, G_losses = [], []
    batch = 0
    for batch_idx in range(train_size//bs):
       
        t_batch = t_train[batch:(batch+bs),:]
        u_batch = u_train[batch:(batch+bs),:]
    
        G_losses.append(G_train(t_batch,u_batch,x_col))
        D_losses.append(D_train(t_batch,u_batch))
        batch += bs

    print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
            (epoch), n_epochs, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))
                                                                                                                    
    
with torch.no_grad():
   
    generated = G(t_test)
    u = generated.cpu().detach().numpy()
    plt.plot(t,sol_plot[0,:],label='Real solutions')
    plt.scatter(t_train,u_train,label = 'Training points',s=8,color='red')
    plt.plot(t,u,label='Generated solutions')
    plt.legend()
    plt.show()

    MSE = np.square(np.subtract(sol_plot[0,:],u[:,0])).mean()
    print(MSE)