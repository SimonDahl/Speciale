# prerequisites
from ast import arg
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
import argparse
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from scipy.integrate import odeint



#%% Hyperparameters

#bs = 5
x_dim =1
y_dim = 1
z_dim = 1

n_neurons = 50

n_data = 20
timesteps = 100
slope = 0.01
drop = 0.2
criterion = nn.BCELoss() 
lr =  1e-4
np.random.seed(2022)
time_limit = 10
beta = 1 # weigt of ODE loss
#n_epochs = args.n_epochs
#z_dim = args.z_dim
n_epochs = 1
col_res = 1000 # collocation point resolution 
col_points = int(time_limit*col_res) # n collocation points

def pend(x, t, m, k):
    x1,x2 = x
    dxdt = [x2, -m*x2 - k*np.sin(x1)]
    return dxdt


t = np.linspace(0, time_limit, timesteps)
   
data = np.zeros((n_data,timesteps))

#x0 = [1,0.5]

m = 1 
k = 2

for i in range(n_data):
    x0 = [np.random.uniform(0,np.pi),np.random.uniform(0,1)]
    #m = np.random.uniform(0.1,2)
    #k = np.random.uniform(3,10)
    sol = odeint(pend, x0, t, args=(m, k))
    data[i,:] = sol[:,0]
print('Data generation complete')


   
#%% Ready data


#train_set = train_set.float()

#train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=bs, shuffle=True)

#print(train_set.shape)




#%% Define Network




class Generator(nn.Module):
    def __init__(self, g_input_dim, g_output_dim):
        super(Generator, self).__init__()       
        self.fc1 = nn.Linear(g_input_dim, n_neurons)
        self.fc2 = nn.Linear(n_neurons, n_neurons)
        self.fc3 = nn.Linear(n_neurons, n_neurons)
        self.fc4 = nn.Linear(n_neurons, n_neurons)
        self.fc5 = nn.Linear(n_neurons, g_output_dim)
    
        self.k = torch.nn.parameter.Parameter(torch.from_numpy(np.array([1])).float())
        self.m = torch.nn.parameter.Parameter(torch.from_numpy(np.array([1])).float())
        
    # forward method
    def forward(self, x,z):
        y = torch.concat((x,z)) 
        y = F.tanh(self.fc1(y)) # leaky relu, with slope angle 
        y = F.tanh(self.fc2(y)) 
        y = F.tanh(self.fc3(y))
        y = F.tanh(self.fc4(y))
        #return torch.tanh(self.fc4(x))
        return self.fc5(y) 

    
class Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, n_neurons)
        self.fc2 = nn.Linear(n_neurons,n_neurons)
        self.fc3 = nn.Linear(n_neurons,n_neurons)
        self.fc4 = nn.Linear(n_neurons,n_neurons)
        self.fc5 = nn.Linear(n_neurons, 1)  # output dim = 1 for binary classification 
    
    # forward method
    def forward(self, x,y):
        d = torch.concat((x,y))
        d = F.tanh(self.fc1(d))
        d = F.dropout(d, drop)
        d = F.tanh(self.fc2(d))
        d = F.dropout(d, drop)
        d = F.tanh(self.fc3(d))
        d = F.dropout(d, drop)
        d = F.tanh(self.fc4(d))
        d = F.dropout(d, drop)
        return torch.sigmoid(self.fc5(d))  # sigmoid for probaility 
        

class Q_net(nn.Module):
    def __init__(self, Q_input_dim,Q_output_dim):
        super(Q_net, self).__init__()
        self.fc1 = nn.Linear(Q_input_dim, n_neurons)
        self.fc2 = nn.Linear(n_neurons,n_neurons)
        self.fc3 = nn.Linear(n_neurons,n_neurons)
        self.fc4 = nn.Linear(n_neurons,Q_output_dim)
    
    def forward(self,x,y):
        q = torch.concat(x,y)
        q = F.tanh(self.fc1(q)) # leaky relu, with slope angle 
        q = F.tanh(self.fc2(q))
        q = F.tanh(self.fc3(q)) 
        return (self.fc4(q))
        
# Physics-Informed residual on the collocation points         
def compute_residuals(x,z):
    u = G(x,z)
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    u_tt = torch.autograd.grad(u_t.sum(), t, create_graph=True)[0]
    r_ode = m*u_tt+k*x
    return r_ode 
 
def get_u(x,z):
    u = G(x,z)
    return u 
   

def compute_G_loss(x_u, y_u, y_u_pred, x_col, y_col_pred, z_u, Z_col):
    
    # get z from q_net encoder
    z_u_Q_net = Q(x_u,y_u_pred)
    z_f_Q_net = Q(x_col,y_col_pred)
    
    # descriminator loss
    y_pred = G(x_u,z_u)
    D_pred = D(x_u,y_pred)
    
    # kl diverenge between data distribution and the model distribution       
    KL = torch.mean(D_pred)
    
    # Entropic regularization 
    log_q = -torch.mean(torch.square(z_u-z_u_Q_net))
    
    # Physics informed loss 
    
    
    
  
# build network
G = Generator(g_input_dim = z_dim+x_dim, g_output_dim = timesteps).to(device)
D = Discriminator(x_dim+y_dim).to(device)
Q = Q_net(x_dim+y_dim,z_dim)

# set optimizer 
G_optimizer = optim.Adam(G.parameters(), lr=lr)
D_optimizer = optim.Adam(D.parameters(), lr=lr)
Q_optimizer = optim.Adam(Q.parameters(), lr=lr)



#%% Define training loops 

