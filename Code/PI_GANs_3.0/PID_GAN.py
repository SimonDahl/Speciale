# Virker til usikkerheds vurdering 

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


#%% Hyperparameters

time_limit = 6 # time limit for ODE solver
timesteps = 200 # number of timesteps in ODE solver solution 
n_col = 1500 # number of collocation points

n_neurons = 50 # number of neruons in hidden layer(s)
lr = 0.001 # learning rate

z_dim = 1 # for dimension of network 
x_dim = 1
y_dim = 1 

criterion = nn.BCELoss() 
criterion_mse = nn.MSELoss()
n_epochs = 4000 # umber of epochs

gen_epoch = 5 # number of G epochs pr D epoch
n_gens = 30 # number of predicted solutions used for uncertanty quantification 
lambda_phy = 1 # physics loss weight 
lambda_q = 0.4 # Q_new weigth 
lambda_val = 0.05 # for phy_prob 

add_noise = True
noise_level = 0.1 # choose ratio of noise Gaussian noise
mu_noise =0.0 # mean noise value

def gaussian_noise(x,mu,std):
    noise = np.random.normal(mu, std, size = x.shape)
    x_noisy = x + noise
    return x_noisy
    

#%% Calculate traning data 
t = np.linspace(0,time_limit,timesteps)

# pendulum parameters
m = 2
k = 5
c = 1
    
def sho(t,y):
    solution = (y[1],(-(c/m)*y[1]-(k/m)*y[0])) # damped harmonic oscillator
    return solution
    
y_init = [3,1]
solution = solve_ivp(sho, [0,timesteps], y0 = y_init, t_eval = t)
sol_data = solution.y[0]
    
sol_plot = np.array([sol_data])  


# idx of training points chosen - int values from t_start to time_limit
idx = [0,3,10,15,80,85,100,102,175,185]
n_data = len(idx)



if add_noise == True:
    std = noise_level * np.std(sol_data)
    noisy_signal = gaussian_noise(sol_data,mu_noise,std) 
    u_data = np.array(noisy_signal)[idx]
else:
    u_data = np.array(sol_data)[idx]



t_data = np.array(t)[idx] # select corresponding time points 


if add_noise == True:
    noise = np.random.normal(0,noise_level,timesteps)
    noisy_signal = sol_data + noise
    u_data = np.array(noisy_signal)[idx]
    #plt.plot(t,noisy_signal)
    #plt.show()






#%% Create torch variables for autodiff 
 
t_data = Variable(torch.from_numpy(t_data).float(), requires_grad=True).to(device)
t_data = t_data.reshape(n_data,1)
u_data = Variable(torch.from_numpy(u_data).float(), requires_grad=True).to(device)
u_data = u_data.reshape(n_data,1)

t_test = t.reshape(timesteps,1)
t_test = Variable(torch.from_numpy(t_test).float(), requires_grad=True).to(device)

col = np.linspace(0, time_limit, n_col) # define collocation points
col = col.reshape(n_col,1)
col = Variable(torch.from_numpy(col).float(), requires_grad=True).to(device)


start = time.time() 

#%% Network definitions 
class Generator(nn.Module):
    def __init__(self, g_input_dim, g_output_dim):
        super(Generator, self).__init__()       
        self.fc1 = nn.Linear(g_input_dim, n_neurons)
        self.fc2 = nn.Linear(n_neurons, n_neurons)
        self.fc3 = nn.Linear(n_neurons, n_neurons)
        self.fc4 = nn.Linear(n_neurons, g_output_dim)
    
    # forward method
    def forward(self,y):
        y = torch.tanh(self.fc1(y)) 
        y = torch.tanh(self.fc2(y))
        y = torch.tanh(self.fc3(y))  
        return self.fc4(y) 

  
class Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, n_neurons)
        self.fc2 = nn.Linear(n_neurons,n_neurons)
        self.fc3 = nn.Linear(n_neurons,n_neurons)
        self.fc4 = nn.Linear(n_neurons, 1)  # output dim = 1 for discriminator
    
    # forward method
    def forward(self, d):
        d = torch.tanh(self.fc1(d))
        d = torch.tanh(self.fc2(d))
        d = torch.tanh(self.fc3(d))
        return ((self.fc4(d))) 
        
class Q_net(nn.Module):
    def __init__(self, Q_input_dim,Q_output_dim):
        super(Q_net, self).__init__()
        self.fc1 = nn.Linear(Q_input_dim, n_neurons)
        self.fc2 = nn.Linear(n_neurons,n_neurons)
        self.fc3 = nn.Linear(n_neurons,n_neurons)
        self.fc4 = nn.Linear(n_neurons,Q_output_dim)
    
    def forward(self,q):
     
        q = torch.tanh(self.fc1(q))
        q = torch.tanh(self.fc2(q))
        q = torch.tanh(self.fc3(q))
        return (self.fc4(q))

# build networks
G = Generator(g_input_dim = z_dim+x_dim, g_output_dim = 1).to(device)
D = Discriminator(x_dim+y_dim+1).to(device) # Plus 1 for eta i.e Phy Prob value
Q = Q_net(x_dim+y_dim,1)

# set optimizer 
G_optimizer = optim.Adam(G.parameters(), lr=lr)
D_optimizer = optim.Adam(D.parameters(), lr=lr)
Q_optimizer = optim.Adam(Q.parameters(), lr=lr)


# Physics-Informed residuals       
def compute_residuals(t,u):

    u_t  = torch.autograd.grad(u, t, torch.ones_like(u), retain_graph=True,create_graph=True)[0]# computes dy/dx
    u_tt = torch.autograd.grad(u_t, t, torch.ones_like(u_t),retain_graph=True ,create_graph=True)[0]# computes d^2y/dx^2
    
    r_ode = m*u_tt+c*u_t + k*u
    return r_ode


def n_phy_prob(t): # function for calculing residuals and for using same z for all networks in same epoch
    z = Variable(torch.randn(t.shape).to(device))
    g_input = torch.cat((t,z),dim=1)
    u_pred = G(g_input) 
    res = compute_residuals(t,u_pred) # compute residuals 
    n_phy = torch.exp(-lambda_val * (res**2)) # compute physics consistency score
    return u_pred,z,res,n_phy



def discriminator_loss(logits_real_u, logits_fake_u, logits_fake_f, logits_real_f):
        loss =  - torch.mean(torch.log(1.0 - torch.sigmoid(logits_real_u) + 1e-8)) - torch.mean(torch.log(torch.sigmoid(logits_fake_u) + 1e-8)) \
                - torch.mean(torch.log(1.0 - torch.sigmoid(logits_real_f) + 1e-8)) - torch.mean(torch.log(torch.sigmoid(logits_fake_f) + 1e-8))
        return loss

def generator_loss(logits_fake_u, logits_fake_f):
    gen_loss = torch.mean(logits_fake_u) + torch.mean(logits_fake_f)
    return gen_loss
    


#%% Discriminator training loop  
    
def D_train(t,u_train):
    
    D_optimizer.zero_grad()
     
    real_prob = torch.ones_like(t)
      
    # real y value for Discriminator  
     
    d_input = torch.cat((t,u_train,real_prob),dim=1)
    real_logits = D(d_input)
            
    # physics loss for boundary point 
    u_pred,_,_,n_phy = n_phy_prob(t)
    fake_logits_u = D(torch.cat((t,u_pred,n_phy),dim=1))
        
    # physics loss for collocation points 
    
    u_col,_,_,n_phy_col = n_phy_prob(col)
    fake_logits_col = D(torch.cat((col,u_col,n_phy_col),dim=1))

    # computing synthetic real logits on collocation points for discriminator loss

    real_prob_col = torch.ones_like(col)
    real_logits_col = D(torch.cat((col,u_col,real_prob_col),dim=1))
    
    D_loss = discriminator_loss(real_logits,fake_logits_u,fake_logits_col,real_logits_col) 
    D_loss.backward(retain_graph=True)
    D_optimizer.step()
    return D_loss.data.item()




def G_train(t):

    for g_epoch in range(gen_epoch):
        
        G.zero_grad()

        #physics loss for collocation points
        u_col,_,phyloss1,n_phy  = n_phy_prob(col)
        fake_logits_col = D(torch.cat((col,u_col,n_phy),dim=1))

        # physics loss for boundary points 
        
        u_pred,G_z,phyloss2,n_phy = n_phy_prob(t)
        fake_logits_u = D(torch.cat((t,u_pred,n_phy),dim=1))

        z_pred = Q(torch.cat((t,u_pred),dim=1))
        mse_loss_z = criterion_mse(z_pred,G_z)

        phy_loss = (phyloss1**2).mean()
                
        adv_loss = generator_loss(fake_logits_u,fake_logits_col)
        
        G_loss =  adv_loss + lambda_q * mse_loss_z  
        G_loss.backward(retain_graph=True)
        G_optimizer.step()
        

    return G_loss



def Q_train(t): # Q_net traning loop 
    
    Q_optimizer.zero_grad()
    Q_z = torch.randn(t.shape).to(device)
    u_pred = G(torch.cat((t,Q_z),dim=1))
    z_pred = Q(torch.cat((t,u_pred),dim=1))
    Q_loss = criterion_mse(z_pred,Q_z)
    Q_loss.backward()
    Q_optimizer.step()
    
    return Q_loss.data.item()


  
#%% 
for epoch in range(1, n_epochs+1):
    D_losses, G_losses,Q_losses = [], [],[]

    D_losses.append(D_train(t_data,u_data))
    G_losses.append(G_train(t_data))
    Q_losses.append(Q_train(t_data))

    print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
            (epoch), n_epochs, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))
                                                                                                                  

end = time.time() 
print("Time elapsed during the calculation:", end - start) 


with torch.no_grad():
    
    u_pred_list = []
    for i in range(n_gens): # generate n_gens predictions for uncertanty quantification 

        z = Variable(torch.randn(t_test.shape).to(device))
        generated = G(torch.cat((t_test,z),dim=1))
        u = generated.cpu().detach().numpy()
        u_pred_list.append(u)
     
    
    u_preds = np.array(u_pred_list)
    u_pred = u_preds.mean(axis=0).flatten() # mean prediction 
    u_preds_var = np.var(u_preds,axis=0).flatten() # variance 
     
    plt.plot(t,sol_data,label='Real solution')
    if add_noise == True:
        plt.scatter(t_data,u_data,color='red',label='Noisy Training points')
        plt.title('Pendulum solution with UQ - '+str(noise_level*100)+'% Noise level' )
    else:
        plt.scatter(t_data,u_data,color='red',label='Training points')
    
    std = 2.0*np.sqrt(u_preds_var) # 2 std band
       
    plt.fill_between(t, u_pred - std, u_pred + std,
                 color='grey', alpha=0.4,label='2 std band') # visualize error 
    plt.plot(t,u_pred,'--',label='Predicted Solution')
    plt.legend(loc = 'upper right')
    plt.xlabel('Time')
    plt.ylabel('Position')
    MSE = (np.square(np.subtract(u_pred,sol_data))).mean() # calculate MSE of predicted solution 
    print('MSE '+str(MSE)) 

    plt.show()

  