# er lavet helt som i artiklen, har både boundary points og tager kun et punkt af gangen som input. ser ikke ud til at lære noget som helst 

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


n_data =100
n_sols = 30
bs = 100
time_limit = 6
n_col = 1000



#y_data = np.cos(x_data*np.sqrt(k)) # Exact solution for (0,1) boundary condition
n_neurons = 50
lr = 0.001
drop = 0.0
z_dim = 1
x_dim = 1
y_dim = 1 
criterion = nn.BCELoss() 
criterion_mse = nn.MSELoss()
n_epochs = 100

gen_epoch = 5
lambda_phy = 1
lambda_q = 0.05
lambda_val = 0.05
#y_data = -k*np.cos()+k
timesteps = 100


t = np.linspace(0,time_limit,timesteps)

#y_b = np.zeros((n_data,1))
#y = [2,1]


#idx = [0,2,3,4,5,6,12,15,21,44,50,55,82,88,89,95,98,101,111,120,127,138,148,150,154,160,175,180,189,198] # index for data points in solution data

idx = list(range(0,100))

m = 2
k = 5
c = 1
    
def sho(t,y):
    m = np.random.uniform(1,5)
    solution = (y[1],(-(c/m)*y[1]-(k/m)*y[0]))
    return solution
    



y_train = np.zeros((n_sols,n_data))
x_train = np.zeros((n_sols,n_data))


x_b = list(t[i] for i in idx)
x_b = np.array(x_b)

for i in range(n_sols):
    y_init = [np.random.uniform(1,10),np.random.uniform(-2,2)]
    solution = solve_ivp(sho, [0,timesteps], y0 = y_init, t_eval = t)
    sol_data = solution.y[0]
    sol_data = list(sol_data[i] for i in idx)
    sol_data = np.array([sol_data])
    y_train[i,:] = sol_data
    x_train[i,:] = x_b

plt.plot(t,y_train[5,:])
plt.show()  
x_col = np.linspace(0, time_limit, n_col)


#y_b = list(y_real[i] for i in idx)

#y_b = np.array([y_b])


t_sample = t.reshape(timesteps,1)
#Xmean, Xstd = x_col.mean(0), x_col.std(0)
#x_col = (x_col - Xmean) / Xstd
#x_b = (x_b - Xmean) / Xstd
#X_star_norm = (t_sample - Xmean) / Xstd 

x_b = Variable(torch.from_numpy(x_train).float(), requires_grad=True).to(device)
y_b = Variable(torch.from_numpy(y_train).float(), requires_grad=True).to(device)
y_b = y_b.reshape((n_sols,n_data))
x_b = x_b.reshape((n_sols,n_data))


X_star_norm = Variable(torch.from_numpy(t_sample).float(), requires_grad=True).to(device)
x_col = x_col.reshape(n_col,1)
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
        d = F.dropout(d, drop)
        d = torch.tanh(self.fc2(d))
        d = F.dropout(d, drop)
              
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

# build network
G = Generator(g_input_dim = z_dim+x_dim, g_output_dim = 1).to(device)
D = Discriminator(x_dim+y_dim).to(device)
Q = Q_net(x_dim+y_dim,1)


# set optimizer 
G_optimizer = optim.Adam(G.parameters(), lr=lr)
D_optimizer = optim.Adam(D.parameters(), lr=lr)
Q_optimizer = optim.Adam(Q.parameters(), lr=lr)


# Physics-Informed residual on the collocation points         
def compute_residuals(x,u):
    
   
    #z = Variable(torch.randn(z_dim).to(device))
               
    u_t  = torch.autograd.grad(u, x, torch.ones_like(u), retain_graph=True,create_graph=True)[0]# computes dy/dx
    u_tt = torch.autograd.grad(u_t,  x, torch.ones_like(u_t),retain_graph=True ,create_graph=True)[0]# computes d^2y/dx^2
       
    r_ode = m*u_tt+c*u_t + k*u# computes the residual of the 1D harmonic oscillator differential equation
    #r_ode = m*u_tt + k*u
     
    #r_ode = k*u-u_t   
    return r_ode


def n_phy_prob(x):
    noise = Variable(torch.randn(x.shape).to(device))
    g_input = torch.cat((x,noise),dim=1)
    u = G(g_input)
    res = compute_residuals(x,u)
    #n_phy = torch.exp(-lambda_val * (res**2))
        
    return u,noise,res

def discriminator_loss(logits_real_u, logits_fake_u):
        loss =  - torch.mean(torch.log(1.0 - torch.sigmoid(logits_real_u) + 1e-8) + torch.log(torch.sigmoid(logits_fake_u) + 1e-8)) 
        return loss

def generator_loss(logits_fake_u):
        gen_loss = torch.mean(logits_fake_u)
        return gen_loss
    
    
    
def D_train(x,y_train):
    
    D_optimizer.zero_grad()
     
         
    # real y value for Discriminator  
    
    D_loss = torch.zeros(bs)
    
            
    for i in range(bs):
        
        x_i = x[i,:]
        y_i = y_train[i,:]
        
      
        x_i = x_i.reshape((n_data,1))
        y_i = y_i.reshape((n_data,1))
        
        
        # real y value for Discriminator  
   
        d_input = torch.cat((x_i,y_i),dim=1)
        real_logits = D(d_input)
    
        
        # physics loss for boundary point 
        u,_,_ = n_phy_prob(x_i)
   
        fake_logits_u = D(torch.cat((x_i,u),dim=1))
    
   
        D_loss[i] = discriminator_loss(real_logits, fake_logits_u)
    
    D_loss = torch.mean(D_loss)
    D_loss.backward(retain_graph=True)
    D_optimizer.step()
    return D_loss.data.item()
        
       
        
    D_loss = torch.mean(D_loss)
    
    D_loss.backward(retain_graph=True)
    D_optimizer.step()
    return D_loss.data.item()


def G_train(x,y_train):

    for g_epoch in range(gen_epoch):
        
        G_loss = torch.zeros(bs)
        
        for i in range(bs):
        
            G.zero_grad()

            #physics loss for collocation points
            
            x_i = x[i,:]
            y_i = y_train[i,:]
        
      
            x_i = x_i.reshape((n_data,1))
            y_i = y_i.reshape((n_data,1))
        
       
            #physics loss for collocation points
        
            # u,noise,n_phy,res
        
            _,_,phyloss1  = n_phy_prob(x_col)
        

          # physics loss for boundary points 
        
            y_pred,G_noise,_ = n_phy_prob(x_i)
            fake_logits_u = D(torch.cat((x_i,y_pred),dim=1))

            z_pred = Q(torch.cat((x_i,y_pred),dim=1))
            mse_loss_z = criterion_mse(z_pred,G_noise)

            mse_loss = criterion_mse(y_pred,y_i)
        
            adv_loss = generator_loss(fake_logits_u)
        
            phy_loss = (phyloss1**2).mean()
        
            G_loss[i] = adv_loss + lambda_phy * phy_loss + lambda_q * mse_loss_z #+ mse_loss 

        
        G_loss = torch.mean(G_loss)
        G_loss.backward(retain_graph=True)
        G_optimizer.step()

        return G_loss


def Q_train(x):
    
    Q_optimizer.zero_grad()
    
    Q_loss = torch.zeros(bs)
    
    for i in range(bs):
        
        
        x_i = x[i,:]
        x_i = x_i.reshape((n_data,1))
        Q_noise = torch.randn(x_i.shape).to(device)
    
        y_pred = G(torch.cat((x_i,Q_noise),dim=1))
        z_pred = Q(torch.cat((x_i,y_pred),dim=1))
        Q_loss[i] = criterion_mse(z_pred,Q_noise)
    
    Q_loss = torch.mean(Q_loss)
    Q_loss.backward()
    Q_optimizer.step()
    
    return Q_loss.data.item()


  
#%% 
for epoch in range(1, n_epochs+1):
    D_losses, G_losses,Q_losses = [], [],[]
    batch = 0
    for i in range((n_sols//bs)):
               
        
             
        x_batch = x_b[batch:(batch+bs),:]
        y_batch = y_b[batch:(batch+bs),:]
        
        D_losses.append(D_train(x_batch,y_batch))
        G_losses.append(G_train(x_batch,y_batch))
        Q_losses.append(Q_train(x_batch))
        batch += bs
       
       
    print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
            (epoch), n_epochs, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))
                                                                                                                    


with torch.no_grad():
    
    
    for i in range(3):
        z = Variable(torch.randn(X_star_norm.shape).to(device))
        generated = G(torch.cat((X_star_norm,z),dim=1))
        y = generated.cpu().detach().numpy()
        plt.plot(t,y)
    
    #plt.plot(t,y_real)
    plt.show()
