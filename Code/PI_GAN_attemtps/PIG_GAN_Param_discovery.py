# Den her virker til en løsning 

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



bs = 1
n_col = 2500
n_neurons = 50
lr = 0.001
drop = 0.0
n_epochs = 4000
gen_epoch = 5


z_dim = 1
x_dim = 1
y_dim = 1 

criterion = nn.BCELoss() 
criterion_mse = nn.MSELoss()



lambda_phy = 0.8
lambda_q = 0.4
lambda_val = 0.05

time_limit = 6
timesteps = 100
t = np.linspace(0,time_limit,timesteps)

#%% Genarate data


m_known = 2
k = 5
c = 1

    
def sho(t,y):
    solution = (y[1],(-(c/m_known)*y[1]-(k/m_known)*y[0]))
    return solution
    

y_init = [3,1]
solution = solve_ivp(sho, [0,timesteps], y0 = y_init, t_eval = t)
sol_data = solution.y[0]
    
sol_plot = np.array([sol_data])  

y_b = [sol_data[0],sol_data[5],sol_data[10],sol_data[32],sol_data[45],sol_data[65],sol_data[73],sol_data[98]]
y_b = np.array([y_b])
  

x_b = [t[0],t[5],t[10],t[32],t[45],t[65],t[73],t[98]]
n_data = len(x_b)

x_b = np.array([x_b])



x_col = np.linspace(0, time_limit, n_col)

t_sample = t.reshape(timesteps,1)
x_b = Variable(torch.from_numpy(x_b).float(), requires_grad=True).to(device)
y_b = Variable(torch.from_numpy(y_b).float(), requires_grad=True).to(device)
y_b = y_b.reshape(n_data,1)
x_b = x_b.reshape(n_data,1)



X_star_norm = Variable(torch.from_numpy(t_sample).float(), requires_grad=True).to(device)
x_col = x_col.reshape(n_col,1)
x_col = Variable(torch.from_numpy(x_col).float(), requires_grad=True).to(device)


start = time.time() 


#%% Define network 


class Generator(nn.Module):
    def __init__(self, g_input_dim, g_output_dim):
        super(Generator, self).__init__()       
        self.fc1 = nn.Linear(g_input_dim, n_neurons)
        self.fc2 = nn.Linear(n_neurons, n_neurons)
        self.fc3 = nn.Linear(n_neurons, g_output_dim)
        
        self.m = torch.nn.parameter.Parameter(torch.from_numpy(np.array([1])).float(), requires_grad=True)
        #self.k = torch.nn.parameter.Parameter(torch.from_numpy(np.array([1])).float(), requires_grad=True)
    
        
        
    # forward method
    def forward(self,y):
        y = torch.tanh(self.fc1(y)) #
        y = torch.tanh(self.fc2(y)) 
       
        return self.fc3(y) 
    
       
    def getODEParam(self):
        return (self.m)
    
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
    
#%% 

# build network
G = Generator(g_input_dim = z_dim+x_dim, g_output_dim = 1).to(device)
D = Discriminator(x_dim+y_dim).to(device)
Q = Q_net(x_dim+y_dim,1)


# set optimizer 
G_optimizer = optim.Adam(G.parameters(), lr=lr)
D_optimizer = optim.Adam(D.parameters(), lr=lr)
Q_optimizer = optim.Adam(Q.parameters(), lr=lr)

#%%


# Physics-Informed residual on the collocation points         
def compute_residuals(x,u,m):           
   
    u_t  = torch.autograd.grad(u, x, torch.ones_like(u), retain_graph=True,create_graph=True)[0]# computes dy/dx
    u_tt = torch.autograd.grad(u_t,  x, torch.ones_like(u_t),retain_graph=True ,create_graph=True)[0]# computes d^2y/dx^2
    
   
    r_ode = m*u_tt+c*u_t + k*u
          
    return r_ode


def n_phy_prob(x):
    noise = Variable(torch.randn(x.shape).to(device))
    g_input = torch.cat((x,noise),dim=1)
    u = G(g_input)
    m = G.getODEParam()
    res = compute_residuals(x,u,m)
           
    return u,noise,res

def discriminator_loss(logits_real_u, logits_fake_u):
        loss =  - torch.mean(torch.log(1.0 - torch.sigmoid(logits_real_u) + 1e-8) + torch.log(torch.sigmoid(logits_fake_u) + 1e-8)) 
        return loss

def generator_loss(logits_fake_u):
        gen_loss = torch.mean(logits_fake_u)
        return gen_loss
    
 
#%% Discriminator training loop  
    
def D_train(x,y_train):
    
    D_optimizer.zero_grad()
     
    
    # real y value for Discriminator  
   
    d_input = torch.cat((x,y_train),dim=1)
    real_logits = D(d_input)
    
        
    # physics loss for boundary point 
    u,_,_ = n_phy_prob(x)
   
    fake_logits_u = D(torch.cat((x,u),dim=1))
    
   
    D_loss = discriminator_loss(real_logits, fake_logits_u)
    D_loss.backward(retain_graph=True)
    D_optimizer.step()
    return D_loss.data.item()


#%% Generator training loop 

def G_train(x,y_train):

    for g_epoch in range(gen_epoch):
        
        G.zero_grad()

        #physics loss for collocation points
        
        # u,noise,n_phy,res
        
        _,_,phyloss1  = n_phy_prob(x_col)
        

        # physics loss for boundary points 
        
        y_pred,G_noise,_ = n_phy_prob(x)
        fake_logits_u = D(torch.cat((x,y_pred),dim=1))

        z_pred = Q(torch.cat((x,y_pred),dim=1))
        mse_loss_z = criterion_mse(z_pred,G_noise)

        mse_loss = criterion_mse(y_pred,y_train)
        
        adv_loss = generator_loss(fake_logits_u)
        
        phy_loss = (phyloss1**2).mean()
        
        G_loss = adv_loss + lambda_phy * phy_loss + lambda_q * mse_loss_z
        
        
        G_loss.backward(retain_graph=True)
        G_optimizer.step()

    m_approx = G.getODEParam()
  #  k_approx = k_approx.detach().numpy()
    m_approx = m_approx.detach().numpy()
    
    return G_loss,m_approx


#%% Q_net train 

def Q_train(x):
    
    Q_optimizer.zero_grad()
    Q_noise = torch.randn(x.shape).to(device)
    
    y_pred = G(torch.cat((x,Q_noise),dim=1))
    z_pred = Q(torch.cat((x,y_pred),dim=1))
    Q_loss = criterion_mse(z_pred,Q_noise)
    Q_loss.backward()
    Q_optimizer.step()
    
    return Q_loss.data.item()




m_approx = np.zeros(n_epochs)
#k_approx = np.zeros(n_epochs)
m_true_list = np.repeat(m_known,n_epochs)
#k_true_list = np.repeat(k_known,n_epochs)

#%% 
for epoch in range(1, n_epochs+1):
    D_losses, G_losses,Q_losses = [], [],[]


    D_losses.append(D_train(x_b,y_b))
    loss,m = G_train(x_b,y_b)
    G_losses.append(loss)
    m_approx[epoch-1] = m
   # k_approx[epoch-1] = k 
    Q_losses.append(Q_train(x_b))

    print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
            (epoch), n_epochs, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))
                                                                                                                    

end = time.time() 
print("Time elapsed during the calculation:", end - start) 


#%% Show generated sample 

with torch.no_grad():
    
    
    for i in range(1):
        z = Variable(torch.randn(X_star_norm.shape).to(device))
        generated = G(torch.cat((X_star_norm,z),dim=1))
        y = generated.cpu().detach().numpy()
        plt.plot(t,y,'--',label='Generated solution')
    plt.plot(t,sol_data,label='Real solution')
    plt.scatter(x_b,y_b,color='red',label='Training points')
    plt.title('Generetated solution with unknown m')
    plt.xlabel('t [s]')
    plt.legend(loc = 'upper right')
    plt.show()

    

e_plot = list(range(n_epochs))
with torch.no_grad():

    plt.plot(e_plot,m_true_list,label='M True',color='red')
    plt.plot(e_plot,m_approx,'--',label='M approx',color='red')
    #plt.plot(e_plot,k_true_list,label='K True',color='blue')
    #plt.plot(e_plot,k_approx,'--',label='K approx',color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Approximate m value')
    plt.legend()
    plt.title('Data driven discovery of parameter m')
    plt.show()
 
print('Approx m value '+str(m_approx[-1]))