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



n_neurons = 75
lr = 0.001 # learing rate
lr2 = 0.0001 # learning rate 2 is switch is used
lr_switch = 200000 # n_epochs before changing lr 
criterion = nn.MSELoss() # loss function 
n_epochs = 10000
n_col = 3000 # number of collocation points 
SoftAdapt_beta = 0.1 # soft adabt hyberparamter 



SoftAdapt_start = 200 # soft adabt start epoch 
n_soft = 10 # n loss epochs used for soft adabt



timesteps = 200 # number of timestpes for solver
time_limit = 6 # solver time limit 

# pendumlum paramters 
m_true = 2
k_true = 5
c = 1

t = np.linspace(0,time_limit,timesteps)
    
def sho(t,y):
    solution = (y[1],(-(c/m_true)*y[1]-(k_true/m_true)*y[0])) # damped harmonic oscillator 
    return solution
    
y_init = [3,0] # initial condition
solution = solve_ivp(sho, [0,timesteps], y0 = y_init, t_eval = t) 
sol_data = solution.y[0]

sol_plot = np.array([sol_data])  


u_b = [sol_data[0],sol_data[20],sol_data[40],sol_data[55],sol_data[70],sol_data[100],sol_data[110],sol_data[140],sol_data[180]]

n_b = len(u_b)
  
u_b = np.array([u_b])
  
  
t_b = [t[0],t[20],t[40],t[55],t[70],t[100],t[110],t[140],t[180]]
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
    
        #self.m = torch.nn.parameter.Parameter(torch.from_numpy(np.array([1])).float())
        self.m = torch.nn.parameter.Parameter(torch.from_numpy(np.array([1])).float(), requires_grad=True)
        self.k = torch.nn.parameter.Parameter(torch.from_numpy(np.array([1])).float(), requires_grad=True)
       
    # forward method
    def forward(self,y):
        y = torch.tanh(self.fc1(y)) 
        y = torch.tanh(self.fc2(y))
        y = torch.tanh(self.fc3(y))
        y = torch.tanh(self.fc4(y)) 
      
        return self.fc5(y) 
    
    def getODEParam(self):
        
        return (self.m,self.k)
  
 
net = PINN().to(device)
    
optimizer = optim.Adam(net.parameters(), lr=lr)


def compute_residuals(x):
    
    m,k = net.getODEParam()
    
    u = net(x) # calculate u
 
    u_t  = torch.autograd.grad(u, x, torch.ones_like(u), retain_graph=True,create_graph=True)[0]# computes du/dx
    u_tt = torch.autograd.grad(u_t,  x, torch.ones_like(u_t),retain_graph=True ,create_graph=True)[0]# computes d^2u/dx^2

    
    r_ode = m*u_tt+c*u_t + k*u # damped harmonic oscillator 
    
     
    return r_ode



def SoftAdapt(MSE_us,MSE_fs):
    eps = 10e-8 # for numeric stability 
    
    s_f = np.zeros(n_soft-1) # allocate s_i - the loss rate of change 
    s_u = np.zeros(n_soft-1)
    
    MSE_u = MSE_us[-n_soft:] # chosse n chosen last losses 
    MSE_f = MSE_fs[-n_soft:]
  
    for i in range(1,(n_soft-1)): # calculate s_i
        s_f[i] = MSE_f[i] - MSE_f[i-1] 
        s_u[i] = MSE_u[i] - MSE_u[i-1] 
            
    Beta = SoftAdapt_beta # beta hyper parameter 
    
    # calculate a_i weigths 
    a_f = (np.exp(Beta*(s_f[-1]-np.max(s_f))))/(np.exp(Beta*(s_f[-1]-np.max(s_f)))+np.exp(Beta*(s_u[-1]-np.max(s_u)))+eps)
    a_u = (np.exp(Beta*(s_u[-1]-np.max(s_u))))/(np.exp(Beta*(s_f[-1]-np.max(s_f)))+np.exp(Beta*(s_u[-1]-np.max(s_u)))+eps)    
    
    return a_u,a_f
        
       
# craete loss lists
MSE_us = []
MSE_fs = []    

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
    loss = MSE_u/n_b + MSE_f /n_col 
    
    MSE_us.append(MSE_u/n_b)
    MSE_fs.append(MSE_f/n_col)
    
    if epoch > SoftAdapt_start: # start soft adabt 
        a_u,a_f =SoftAdapt(MSE_us,MSE_fs)
        loss = a_u * MSE_u + a_f *MSE_f
        
    loss.backward()
    
    optimizer.step()
    
    m_approx,k_approx = net.getODEParam()
    k_approx = k_approx.detach().numpy()
    m_approx = m_approx.detach().numpy()
    
    return loss,m_approx,k_approx

losses = []
m_approx = np.zeros(n_epochs)
k_approx = np.zeros(n_epochs)
m_true_list = np.repeat(m_true,n_epochs)
k_true_list = np.repeat(k_true,n_epochs)

for epoch in range(1, n_epochs+1):
    

    
    if epoch > lr_switch: # learning rate switz if desired 
        optimizer = optim.Adam(net.parameters(), lr=lr2)
    
    loss,m,k = train(x_col,u_b,epoch)
    
    losses.append(loss)
    m_approx[epoch-1] = m
    k_approx[epoch-1] = k
    print('[%d/%d]: loss: %.4f' % ((epoch), n_epochs, torch.mean(torch.FloatTensor(losses))))
    

stop = time.time()

#print('Time ussage',stop-start)


with torch.no_grad():
    
    y = net(t_plot) # get final approximation from PINN 
     
    plt.plot(t,sol_data,label='Real solution')
    plt.scatter(t_b,u_b,color='red',label='Data points')
    plt.plot(t,y,'--',label='PINN solution')
    plt.title('Damped Harmonic Oscillator unkown m,k')
    plt.legend()
    plt.show()
     
e_plot = list(range(n_epochs))
with torch.no_grad():

    plt.plot(e_plot,m_true_list,label='M True',color='red')
    plt.plot(e_plot,m_approx,'--',label='M approx',color='red')
    plt.plot(e_plot,k_true_list,label='K True',color='blue')
    plt.plot(e_plot,k_approx,'--',label='K approx',color='blue')
    plt.legend()
    plt.title('Data driven discovery of parameters m and k')
    plt.show()
 
 
 
"""
plt.plot(e_plot,losses)
plt.yscale('log')
plt.title('Loss vs epoch (y log scale)')
plt.show()

with torch.no_grad():

    plt.plot(e_plot,MSE_us,label='MSE_u')
    plt.plot(e_plot,MSE_fs,label='MSE_f')
    plt.yscale('log')
    plt.legend()
    plt.title('MSE_f and MSE_u losses vs epoch (y log scale)')
    plt.show() """
