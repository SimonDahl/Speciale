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



n_neurons = 30
lr = 0.001 # learing rate
lr2 = 0.0001 # learning rate 2 is switch is used
lr_switch = 80000 # n_epochs before changing lr 
criterion = nn.MSELoss() # loss function 
n_epochs = 1000
n_col = 1000 # number of collocation points 
SoftAdapt_beta = 0.1 # soft adabt hyberparamter 



SoftAdapt_start = 2000 # soft adabt start epoch 
n_soft = 10 # n loss epochs used for soft adabt



timesteps = 400 # number of timestpes for solver
time_limit = 10 # solver time limit 

# pendumlum paramters 
m = 2
k = 5
c = 1

t = np.linspace(0,time_limit,timesteps)
    


idx = [0,3,8,15,20,30,38,45,70,75,80,102,122,130,145,158,180,185,195,215,230,240,250,275,300,325,350,375,390]
sol_data = np.sin(t)
sol_dot = np.cos(t)
sol_dotdot = -np.sin(t)

# u_dot = cos
# u_dot_dot = -sin 



u_b = np.array(sol_data)[idx]

n_b = len(u_b)
 
  
t_b = np.array(t)[idx]





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
       # y = torch.tanh(self.fc4(y)) 
      
        return self.fc5(y) 
  
 
net = PINN().to(device)
    
optimizer = optim.Adam(net.parameters(), lr=lr)


def compute_residuals(x):
    
    u = net(x) # calculate u

 
    u_t  = torch.autograd.grad(u, x, torch.ones_like(u), retain_graph=True,create_graph=True)[0]# computes du/dx
    u_tt = torch.autograd.grad(u_t,  x, torch.ones_like(u_t),retain_graph=True ,create_graph=True)[0]# computes d^2u/dx^2

    
    res_1 = u_t - torch.cos(x)
    res_2 = u_tt + torch.sin(x)
         
    return res_1 + res_2


       

start = time.time()


def train(x_col,u_b,epoch):
    optimizer.zero_grad()
    
    # boundary/data points  loss 
    net_u_b = net(t_b)
    MSE_u = criterion(net_u_b,u_b)
    
    # collocation loss 
    
    res = compute_residuals(x_col)
    col_target = torch.zeros_like(res)
    
    MSE_f = criterion(res,col_target)
    # loss normlaized to amount of poins 
    loss = MSE_u + MSE_f  
    
   
    loss.backward()
    
    optimizer.step()
    
    return loss.data.item()

losses = []


for epoch in range(1, n_epochs+1):
    
    
    if epoch > lr_switch: # learning rate switz if desired 
        optimizer = optim.Adam(net.parameters(), lr=lr2)
    
    losses.append(train(x_col,u_b,epoch))

    print('[%d/%d]: loss: %.4f' % ((epoch), n_epochs, torch.mean(torch.FloatTensor(losses))))
    

stop = time.time()

print('Time ussage',stop-start)

with torch.no_grad():
    
    y = net(t_plot) # get final approximation from PINN 
     
    plt.plot(t,sol_data,label='Real solution')
    plt.scatter(t_b,u_b,color='red',label='Training points')
    plt.plot(t,y,'--',label='PINN solution')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.show()

  
    MSE = np.square(np.subtract(y.detach().numpy()[:,0],sol_data)).mean()
    print('MSE loss '+ str(MSE))

 
 

y_test = net(t_plot)
u_t  = torch.autograd.grad(y_test, t_plot, torch.ones_like(y_test), retain_graph=True,create_graph=True)[0]# computes du/dx
u_tt  = torch.autograd.grad(u_t, t_plot, torch.ones_like(y_test), retain_graph=True,create_graph=True)[0]# computes du/dx
SEdot = np.square(np.subtract(u_t.detach().numpy()[:,0],sol_dot))
MSEdot = SEdot.mean()
SEdotdot = np.square(np.subtract(u_tt.detach().numpy()[:,0],sol_dotdot))
MSEdotdot = SEdotdot.mean()

plt.plot(t,SEdot,label='$\dot{u}$ residual')
plt.plot(t,SEdotdot,label='$\ddot{u}$ residual')
plt.yscale('log')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Squared Error')
plt.show()
print("MSE dot "+str(MSEdot))
print("MSE dotdot "+str(MSEdotdot))

e_plot = list(range(n_epochs))

plt.plot(e_plot,losses)
plt.yscale('log')
plt.title('Loss vs epoch (y log scale)')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.show()



