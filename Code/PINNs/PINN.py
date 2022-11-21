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
from scipy.integrate import odeint, solve_ivp


n_neurons = 50
lr = 0.001
criterion = nn.MSELoss()
n_epochs = 7000
n_col = 3000

timesteps = 200
time_limit = 6

t = np.linspace(0,time_limit,timesteps)
t_b = np.array([0])

m = 2
k = 5
c = 1
    
def sho(t,y):
    solution = (y[1],(-(c/m)*y[1]-(k/m)*y[0]))
    return solution
    
y_init = [3,1]
solution = solve_ivp(sho, [0,timesteps], y0 = y_init, t_eval = t)
sol_data = solution.y[0]

sol_plot = np.array([sol_data])  


u_b = [sol_data[0],sol_data[5],sol_data[40],sol_data[100],sol_data[150],sol_data[198]]

n_b = len(u_b)
  
u_b = np.array([u_b])
  
  
t_b = [t[0],t[5],t[40],t[100],t[150],t[198]]
t_b = np.array([t_b])




x_col = np.linspace(0, time_limit, n_col)
x_col = Variable(torch.from_numpy(x_col).float(), requires_grad=True).to(device)
t_b = Variable(torch.from_numpy(t_b).float(), requires_grad=True).to(device)
u_b = Variable(torch.from_numpy(u_b).float(), requires_grad=True).to(device)

t_plot = Variable(torch.from_numpy(t).float(), requires_grad=True).to(device)
t_plot = t_plot.reshape(timesteps,1)

u_b = u_b.reshape(n_b,1)
t_b = t_b.reshape(n_b,1)
x_col = x_col.reshape(n_col,1)

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()       
        self.fc1 = nn.Linear(1, n_neurons)
        self.fc2 = nn.Linear(n_neurons, n_neurons)
        self.fc3 = nn.Linear(n_neurons,n_neurons)
        self.fc4 = nn.Linear(n_neurons,n_neurons)
        self.fc5 = nn.Linear(n_neurons,1)
        
       
    # forward method
    def forward(self,y):
        y = torch.tanh(self.fc1(y)) 
        y = torch.tanh(self.fc2(y))
        y = torch.tanh(self.fc3(y))
        y = torch.tanh(self.fc4(y)) 
      
        return self.fc5(y) 
  
    
   
net = PINN().to(device)
optimizer = optim.Adam(net.parameters(), lr=lr)

def compute_residuals(x):
    #z = Variable(torch.randn(z_dim).to(device))

    u = net(x)
                  
    u_t  = torch.autograd.grad(u, x, torch.ones_like(u), retain_graph=True,create_graph=True)[0]# computes dy/dx
    u_tt = torch.autograd.grad(u_t,  x, torch.ones_like(u_t),retain_graph=True ,create_graph=True)[0]# computes d^2y/dx^2

    r_ode = m*u_tt+c*u_t + k*u
           
    return r_ode


def train(x_col,u_b):
    optimizer.zero_grad()
    
    # boundary loss 
    net_u_b = net(t_b)
    MSE_u = criterion(net_u_b,u_b)
    
    # collocation loss 
    
    res = compute_residuals(x_col)
    col_target = torch.zeros_like(res)
    
    MSE_f = criterion(res,col_target)
    
    loss = MSE_u/n_b + MSE_f /n_col
    
    loss.backward()
    
    optimizer.step()
    
    return loss.data.item()
losses = []
for epoch in range(1, n_epochs+1):
    
    losses.append(train(x_col,u_b))

    print('[%d/%d]: loss: %.4f' % ((epoch), n_epochs, torch.mean(torch.FloatTensor(losses))))
    

with torch.no_grad():
    
    y = net(t_plot)
     
    plt.plot(t,sol_data,label='Real solution')
    plt.scatter(t_b,u_b,color='red',label='Data points')
    plt.plot(t,y,'--',label='PINN solution')
    plt.title('Damped Harmonic Oscillator')
    plt.legend()
    plt.show()


e_plot = list(range(n_epochs))

plt.plot(e_plot,losses)
plt.yscale('log')
plt.title('Loss vs epoch (y log scale)')
plt.show()