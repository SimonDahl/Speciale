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
lr_switch = 2500 # n_epochs before changing lr 
criterion = nn.MSELoss() # loss function 
n_epochs = 20000
n_col = 3000 # number of collocation points 
SoftAdapt_beta = 0.1 # soft adabt hyberparamter 



SoftAdapt_start = 2000 # soft adabt start epoch 
n_soft = 10 # n loss epochs used for soft adabt



timesteps = 200 # number of timestpes for solver
time_limit = 6 # solver time limit 

# pendumlum paramters 
m = 2
k = 5
c = 1

t = np.linspace(0,time_limit,timesteps)
    
def sho(t,y):
    solution = (y[1],(-(c/m)*y[1]-(k/m)*y[0])) # damped harmonic oscillator 
    return solution
    
y_init = [3,0] # initial condition
solution = solve_ivp(sho, [0,timesteps], y0 = y_init, t_eval = t) 
sol_data = solution.y[0]
sol_data_dot = solution.y[1]


sol_plot = np.array([sol_data])  
sol_dot = np.array([sol_data_dot])



u_b = [sol_data[0]]
u_b_dot = [sol_data_dot[0]]



n_b = len(u_b)
  
u_b = np.array([u_b])
u_b_dot = np.array([u_b_dot])
  
t_b = [t[0]]
t_b = np.array([t_b])



""" plt.plot(t,sol_plot[-1,:],label='u')
plt.plot(t,sol_dot[-1,:],label='dot')
plt.scatter(t_b,u_b_dot,label='dot')
plt.scatter(t_b,u_b,label='u')
plt.legend()
plt.show() """


x_col = np.linspace(0, time_limit, n_col)
x_col = Variable(torch.from_numpy(x_col).float(), requires_grad=True).to(device)
t_b = Variable(torch.from_numpy(t_b).float(), requires_grad=True).to(device)
u_b = Variable(torch.from_numpy(u_b).float(), requires_grad=True).to(device)
u_b_dot = Variable(torch.from_numpy(u_b_dot).float(), requires_grad=True).to(device)
t_plot = Variable(torch.from_numpy(t).float(), requires_grad=True).to(device)
t_plot = t_plot.reshape(timesteps,1)

u_b = u_b.reshape(n_b,1)
u_b_dot = u_b_dot.reshape(n_b,1)
t_b = t_b.reshape(n_b,1)
x_col = x_col.reshape(n_col,1)


class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()       
        self.fc1 = nn.Linear(1, n_neurons)
        self.fc2 = nn.Linear(n_neurons, n_neurons)
        self.fc3 = nn.Linear(n_neurons,n_neurons)
        self.fc4 = nn.Linear(n_neurons,n_neurons)
        self.fc5 = nn.Linear(n_neurons,2)
        
       
    # forward method
    def forward(self,y):
        y = F.softplus(self.fc1(y)) 
        y = F.softplus(self.fc2(y))
        y = F.softplus(self.fc3(y))
        y = F.softplus(self.fc4(y)) 
      
        return self.fc5(y) 
  
 
net = PINN().to(device)
    
optimizer = optim.Adam(net.parameters(), lr=lr)



def real_sol(t):
    t4 = torch.tensor(4)
    t39 = torch.tensor(39)
    sol = (torch.exp(-t/t4)*(torch.sqrt(t39)*torch.sin((torch.sqrt(t39)*t)/t4)+t39*torch.cos((torch.sqrt(t39)*t)/t4)))/13
    sol_dot = -(torch.exp(-t/t4)*(torch.sqrt(t39)*torch.sin(torch.sqrt(t39)*t/t4)+t39*torch.cos(torch.sqrt(t39)*t/t4)))/52 \
         + (torch.exp(-t/t4)*((t39*torch.cos(torch.sqrt(t39)*t/t4)/t4)-(t39*torch.sqrt(t39)*torch.sin(torch.sqrt(t39)*t/t4)/t4)))/13
    return sol,sol_dot


""" pp,pp2 = real_sol(x_col.detach().numpy())
plt.plot(x_col.detach().numpy(),pp)
plt.plot(x_col.detach().numpy(),pp2)
plt.show()
 """
def compute_residuals(x):
    
    output = net(x) # calculate u
    x1 = output[:,0]
    x2 = output[:,1]

    real,real_dot =  real_sol(x)
    


    x1 = x1.reshape(n_col,1)
    x2 = x2.reshape(n_col,1)

   
    
        #  solution = (y[1],(-(c/m)*y[1]-(k/m)*y[0]))
                  
    x1_t  = torch.autograd.grad(x1, x, torch.ones_like(x), retain_graph=True,create_graph=True)[0]# computes du/dx
    x2_t = torch.autograd.grad(x2,  x, torch.ones_like(x),retain_graph=True ,create_graph=True)[0]# computes d^2u/dx^2
   
    res1 = x2-x1_t
    res2 = ((-c/m)*x2-(k/m)*x1) -x2_t

    #res1 = real -x1
    #res2 = real_dot-x2

    return res1,res2 



def SoftAdapt(MSE_us,MSE_f_1s,MSE_f_2s):
    eps = 10e-8 # for numeric stability 
    
    s_f_1 = np.zeros(n_soft-1) # allocate s_i - the loss rate of change 
    s_u = np.zeros(n_soft-1)
    s_f_2 = np.zeros(n_soft-1)
    
    
    MSE_u = MSE_us[-n_soft:] # chosse n chosen last losses 
    MSE_f_1 = MSE_f_1s[-n_soft:]
    MSE_f_2 = MSE_f_2s[-n_soft:]
  
    for i in range(1,(n_soft-1)): # calculate s_i
        s_f_1[i] = MSE_f_1[i] - MSE_f_1[i-1] 
        s_u[i] = MSE_u[i] - MSE_u[i-1]
        s_f_2[i] = MSE_f_2[i] - MSE_f_2[i-1] 
            
    Beta = SoftAdapt_beta # beta hyper parameter 
    
    # calculate a_i weigths 
    
    demoninator = (np.exp(Beta*(s_f_1[-1]-np.max(s_f_1)))+np.exp(Beta*(s_u[-1]-np.max(s_u)))+np.exp(Beta*(s_f_2[-1]-np.max(s_f_2))+eps))
    #demoninator = (np.exp(Beta*(s_f[-1]-np.max(s_f)))+np.exp(Beta*(s_u[-1]-np.max(s_u)))+eps)
    a_f_1 = (np.exp(Beta*(s_f_1[-1]-np.max(s_f_1))))/demoninator
    a_u = (np.exp(Beta*(s_u[-1]-np.max(s_u))))/demoninator
    a_f_2 = (np.exp(Beta*(s_f_2[-1]-np.max(s_f_2))))/demoninator
    
    return a_u,a_f_1,a_f_2
        
       
# craete loss lists
MSE_us = []
MSE_f_1s = []    
MSE_f_2s = []

start = time.time()


def train(x_col,u_b,u_b_dot,epoch):
    optimizer.zero_grad()
    
    # boundary/data points  loss 
   
    output = net(t_b)
    #net_u_b = output[:,0]
    #net_u_b_dot = output[:,1]
  
    x1 = output[:,0]
    x2 = output[:,1]
    
    MSE_u = criterion(x1,u_b) + criterion(x2,u_b_dot)
    
    # collocation loss 
    
    res1,res2 = compute_residuals(x_col)
    #print(res1)
    #print(res2)
    col_target = torch.zeros_like(res1)
    
    MSE_f_1 = criterion(res1,col_target)
    MSE_f_2 = criterion(res2,col_target)
    # loss normlaized to amount of poins 
    loss = MSE_f_1  + MSE_f_2 + MSE_u
    
   # MSE_us.append(MSE_u/n_b)
    MSE_f_1s.append(MSE_f_1)
    MSE_f_2s.append(MSE_f_2)
    MSE_us.append(MSE_u)
    

    if epoch > SoftAdapt_start: # start soft adabt 
        a_u,a_f_1,a_f_2 =SoftAdapt(MSE_us,MSE_f_1s,MSE_f_2s)
        loss = a_u * MSE_u + a_f_1 *MSE_f_1 + a_f_2 * MSE_f_2
        
    loss.backward()
    
    optimizer.step()
    
    return loss.data.item()

losses = []


for epoch in range(1, n_epochs+1):
    

    
    if epoch > lr_switch: # learning rate switz if desired 
        optimizer = optim.Adam(net.parameters(), lr=lr2)
    
    losses.append(train(x_col,u_b,u_b_dot,epoch))

    print('[%d/%d]: loss: %.4f' % ((epoch), n_epochs, torch.mean(torch.FloatTensor(losses))))
    

stop = time.time()

print('Time ussage',stop-start)

with torch.no_grad():
    
    output = net(t_plot) # get final approximation from PINN 
    y = output[:,0] 
    plt.plot(t,sol_data,label='Real solution')
    plt.scatter(t_b,u_b,color='red',label='Data points')
    plt.plot(t,y,'--',label='PINN solution')
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.legend()
    MSE = np.square(np.subtract(y.detach().numpy(),sol_data)).mean()
    print('MSE '+str(MSE))
    plt.show()


e_plot = list(range(n_epochs))

plt.plot(e_plot,losses)
plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.show()


