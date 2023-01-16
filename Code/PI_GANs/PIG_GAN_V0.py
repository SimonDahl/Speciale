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


n_sols = 100
bs = 50
time_limit = 6
n_col = 500



#y_data = np.cos(x_data*np.sqrt(k)) # Exact solution for (0,1) boundary condition
n_neurons = 100
lr = 0.001
z_dim = 1
x_dim = 1
y_dim = 1 
criterion = nn.BCELoss() 
criterion_mse = nn.MSELoss()
n_epochs = 500

gen_epoch = 5
lambda_phy = 1
lambda_q = 0.5
lambda_val = 0.05
#y_data = -k*np.cos()+k
timesteps = 200


t = np.linspace(0,time_limit,timesteps)

#y_b = np.zeros((n_data,1))
#y = [2,1]


m = 2
k = 5
c = 1


def sho(t,y):
    solution = (y[1],(-(c/m)*y[1]-(k/m)*y[0]))
    return solution


y_train = np.zeros((n_sols,timesteps))
x_train = np.zeros((n_sols,timesteps))


train = np.zeros((n_sols,timesteps*2))

for i in range(n_sols):
    y_init = [np.random.uniform(1,5),1]
    solution = solve_ivp(sho, [0,timesteps], y0 = y_init, t_eval = t)
    sol_data = solution.y[0]
    train[i,:timesteps] = t
    train[i,timesteps:] = sol_data


t_plot = Variable(torch.from_numpy(t).float(), requires_grad=True).to(device)

#plt.plot(train[0,:timesteps],train[0,timesteps:])

#train_set = torch.from_numpy(train)



#train_set = train_set.float()

train_set =  Variable(torch.from_numpy(train).float(), requires_grad=True).to(device)

train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=bs, shuffle=True)


  
x_col = np.linspace(0, time_limit, timesteps)
x_col = Variable(torch.from_numpy(x_col).float(), requires_grad=True).to(device)

class Generator(nn.Module):
    def __init__(self, g_input_dim, g_output_dim):
        super(Generator, self).__init__()       
        self.fc1 = nn.Linear(g_input_dim, n_neurons)
        self.fc2 = nn.Linear(n_neurons, n_neurons)
        self.fc3 = nn.Linear(n_neurons, g_output_dim)
            
        
    # forward method
    def forward(self,y):
        y = torch.tanh(self.fc1(y)) # leaky relu, with slope angle 
        y = torch.tanh(self.fc2(y)) 
        #return torch.tanh(self.fc4(x))
        return self.fc3(y) 

    
class Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, n_neurons)
        self.fc2 = nn.Linear(n_neurons,n_neurons)
        self.fc3 = nn.Linear(n_neurons, 1)  # output dim = 1 for binary classification 
    
    # forward method
    def forward(self, d):
        d = torch.tanh(self.fc1(d))
        d = torch.tanh(self.fc2(d))
        return ((self.fc3(d))) 
        

class Q_net(nn.Module):
    def __init__(self, Q_input_dim,Q_output_dim):
        super(Q_net, self).__init__()
        self.fc1 = nn.Linear(Q_input_dim, n_neurons)
        self.fc2 = nn.Linear(n_neurons,n_neurons)
        self.fc3 = nn.Linear(n_neurons,Q_output_dim)
    
    def forward(self,q):
     
        q = torch.tanh(self.fc1(q)) # leaky relu, with slope angle 
        q = torch.tanh(self.fc2(q))
        
        return (self.fc3(q))
    
G = Generator(g_input_dim = timesteps*2, g_output_dim = timesteps).to(device)
D = Discriminator(timesteps*2).to(device)
Q = Q_net(timesteps*2,timesteps)

# set optimizer 
G_optimizer = optim.RMSprop(G.parameters(), lr=lr)
D_optimizer = optim.RMSprop(D.parameters(), lr=lr)
Q_optimizer = optim.RMSprop(Q.parameters(), lr=lr)


def compute_residuals(x,u):
    
   
    #z = Variable(torch.randn(z_dim).to(device))
               
    u_t  = torch.autograd.grad(u, x, torch.ones_like(u), retain_graph=True,create_graph=True)[0]# computes dy/dx
    u_tt = torch.autograd.grad(u_t,  x, torch.ones_like(u_t),retain_graph=True ,create_graph=True)[0]# computes d^2y/dx^2
       
    r_ode = m*u_tt+c*u_t + k*u# computes the residual of the 1D harmonic oscillator differential equation
    #r_ode = m*u_tt + k*u
     
    #r_ode = k*u-u_t   
    return r_ode


def n_phy_prob(x,flag):
    if flag == 0:
        noise = Variable(torch.randn(bs, timesteps).to(device))
        g_input = torch.cat((x,noise))
        g_input = g_input.reshape(bs,timesteps*2)
    if flag == 1:
        noise = Variable(torch.randn_like(x))
        g_input = torch.cat((x,noise))
        g_input = g_input.reshape(1,timesteps*2)

    u = G(g_input)
    res = compute_residuals(x,u)
    
        
    return u,noise,res


def discriminator_loss(logits_real_u, logits_fake_u):
        loss =  - torch.mean(torch.log(1.0 - torch.sigmoid(logits_real_u) + 1e-8) + torch.log(torch.sigmoid(logits_fake_u) + 1e-8)) 
        return loss

def generator_loss(logits_fake_u):
        gen_loss = torch.mean(logits_fake_u)
        return gen_loss
    

 
def G_train(x,y):
    
    for g_epoch in range(gen_epoch):
    
        G_optimizer.zero_grad()

        _,_,phyloss  = n_phy_prob(x_col,1) # 1 flag for collocation 


        y_pred,G_noise,_ = n_phy_prob(x,0)
        D_input = torch.cat((x,y_pred))
        D_input = D_input.reshape(bs,timesteps*2)
        
        fake_logits_u = D(D_input)

        Q_input = torch.cat((x,y_pred))
        Q_input = Q_input.reshape(bs,timesteps*2)
        z_pred = Q(Q_input)
        
        mse_loss_z = criterion_mse(z_pred,G_noise)

        mse_loss = criterion_mse(y_pred,y)
        
        adv_loss = generator_loss(fake_logits_u)
        
        phy_loss = (phyloss**2).mean()

        G_loss = adv_loss + lambda_phy * phy_loss + lambda_q * mse_loss_z
        
        
        G_loss.backward(retain_graph=True)
        G_optimizer.step()
        
    return G_loss




def D_train(x,y):
    #=======================Train the discriminator=======================#
    D_optimizer.zero_grad()
    
    D_input = torch.cat((x,y))
    D_input = D_input.reshape(bs,timesteps*2)
        
    real_logits = D(D_input)

        # Collocation points 
        
    u,_,_ = n_phy_prob(x,0)
    D_input = torch.cat((x,u))
    D_input = D_input.reshape(bs,timesteps*2)
        
    fake_logits_u = D(D_input)
        
    D_loss = discriminator_loss(real_logits, fake_logits_u)
    D_loss.backward(retain_graph=True)
    D_optimizer.step()
    return D_loss


def Q_train(x):
    
    Q_optimizer.zero_grad()
    Q_noise = torch.randn(x.shape).to(device)
    G_input = torch.cat((x,Q_noise))
    G_input = G_input.reshape(bs,timesteps*2)
    y_pred = G(G_input)
    
    
    Q_input = torch.cat((x,y_pred))
    Q_input = Q_input.reshape(bs,timesteps*2)
    
    z_pred = Q(Q_input)
    Q_loss = criterion_mse(z_pred,Q_noise)
    Q_loss.backward()
    Q_optimizer.step()
    
    return Q_loss





for epoch in range(1, n_epochs+1):           
    D_losses, G_losses,Q_losses = [], [],[]
    for batch_idx,data in enumerate(train_loader):
        x = data[:,:timesteps]
        y = data[:,timesteps:]
        D_losses.append(D_train(x,y))
        G_losses.append(G_train(x,y))
        Q_losses.append(Q_train(x))

    print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
            (epoch), n_epochs, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))
    

with torch.no_grad():
    
    
    for i in range(5):
        z = Variable(torch.randn(timesteps).to(device))
        G_input = torch.cat((t_plot,z))
        generated = G(G_input)
        y = generated.cpu().detach().numpy()
        plt.plot(t,y)
    
    #plt.plot(t,y_real)
    plt.show()
