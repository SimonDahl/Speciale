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
import warnings
warnings.filterwarnings('ignore')

#%% Hyperparameters
time_limit = 10


bs = 5
x_dim =1
y_dim = 1
z_dim = 1

lam = 1.5
beta = 1

n_col = 100
n_u = 1
n_ref = 200

m = 1 
k = 2


x_col = np.linspace(0,time_limit,n_col)
x_col = torch.tensor(x_col).float()
x_col = x_col.reshape(n_col,-1)  
x_ut = np.array([0.0])

x_ut = torch.tensor(x_ut,requires_grad=True).float()

""" x_u = x_ut 
for i in range(n_u-1):
    x_u = np.vstack((x_u,x_ut)) """

x_u = np.array([0.0])
x_u = torch.tensor(x_u,requires_grad=True).float()

def f(x):
    return np.cos(x*np.sqrt(k))
    
# generate boundary condition 

y_u = f(0.0)
y_u =torch.tensor(y_u,requires_grad=True).float()


""" for i in range(n_u-1):
    y_u = np.vstack((y_u,y_ut))    

y_u = torch.tensor(y_u,requires_grad=True).float()

 """
n_neurons = 50

n_data = 1
timesteps = 100
slope = 0.01
drop = 0.2
criterion = nn.BCELoss() 
lr =  1e-4
#np.random.seed(2022)

beta = 1 # weigt of ODE loss
#n_epochs = args.n_epochs
#z_dim = args.z_dim
n_epochs = 1


def pend(x, t, m, k):
    x1,x2 = x
    dxdt = [x2, -m*x2 - k*np.sin(x1)]
    return dxdt


t = np.linspace(0, time_limit, timesteps)

data = np.zeros((n_data,timesteps))



#x0 = [1,0.5]

for i in range(n_data):
    x0 = [1.0,0.0]
    #m = np.random.uniform(0.1,2)
    #k = np.random.uniform(3,10)
    sol = odeint(pend, x0, t, args=(m, k))
    data[i,:] = sol[:,0]
#print('Data generation complete')

data = data.flatten()

t = Variable(torch.from_numpy(t).float(), requires_grad=True).to(device)     
#%% Ready data

train_set = torch.from_numpy(data)

train_set = train_set.float()

train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=bs, shuffle=True)

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
    def forward(self,y):
        y = torch.tanh(self.fc1(y)) # leaky relu, with slope angle 
        y = torch.tanh(self.fc2(y)) 
        y = torch.tanh(self.fc3(y))
        y = torch.tanh(self.fc4(y))
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
    def forward(self, d):
        d = torch.tanh(self.fc1(d))
        d = F.dropout(d, drop)
        d = torch.tanh(self.fc2(d))
        d = F.dropout(d, drop)
        d = torch.tanh(self.fc3(d))
        d = F.dropout(d, drop)
        d = torch.tanh(self.fc4(d))
        d = F.dropout(d, drop)
        return (self.fc5(d))  # sigmoid for probaility 
        

class Q_net(nn.Module):
    def __init__(self, Q_input_dim,Q_output_dim):
        super(Q_net, self).__init__()
        self.fc1 = nn.Linear(Q_input_dim, n_neurons)
        self.fc2 = nn.Linear(n_neurons,n_neurons)
        self.fc3 = nn.Linear(n_neurons,n_neurons)
        self.fc4 = nn.Linear(n_neurons,Q_output_dim)
    
    def forward(self,q):
     
        q = torch.tanh(self.fc1(q)) # leaky relu, with slope angle 
        q = torch.tanh(self.fc2(q))
        q = torch.tanh(self.fc3(q)) 
        return (self.fc4(q))
        
# Physics-Informed residual on the collocation points         
def compute_residuals(x,z):
    u = G(torch.stack((x,z)))
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    u_tt = torch.autograd.grad(u_t.sum(), t, create_graph=True)[0]
    r_ode = m*u_tt+k*x
    return r_ode 
 
def get_u(x,z):
    u = G(torch.concat(x,z))
    return u 
   

def compute_G_loss(x_u, y_u_pred, x_col, y_col_pred, z_u, z_col):
    
    # get z from q_net encoder
    z_u_Q_net = Q(torch.concat((x_u,y_u_pred)))
    
    
    # descriminator loss
    y_pred = G(torch.concat((x_u,z_u)))
    D_pred = D(torch.concat((x_u,y_pred)))
    
    # kl diverenge between data distribution and the model distribution       
    KL = torch.mean(D_pred)
    
    # Entropic regularization 
    log_q = -torch.mean(torch.square(z_u-z_u_Q_net))
    
    # Physics informed loss 
    
    loss_col = torch.mean(torch.square(y_col_pred))
    
    # generator loss 
    
    loss = KL + (1.0-lam)*log_q + beta * loss_col
    
    return loss, KL, (1.0-lam)*log_q, beta*loss_col 
    
def compute_D_loss(x,y,z):
    
    y_pred = G(torch.concat((x,z)))
    
    # Decriminator loss 
    D_real = D(torch.concat((x,y)))
    D_fake = D(torch.concat((x,y_pred)))
    
    D_real = torch.sigmoid(D_real)
    D_fake = torch.sigmoid(D_fake)
    
    D_loss = -torch.mean(torch.log(1-D_real+1e-8)+torch.log(D_fake+1e-8))
    
    return D_loss

    
  
# build network
G = Generator(g_input_dim = z_dim+x_dim, g_output_dim = 1).to(device)
D = Discriminator(x_dim+y_dim).to(device)
Q = Q_net(x_dim+y_dim,z_dim)

# set optimizer 
#G_optimizer = optim.Adam(G.parameters(), lr=lr)
D_optimizer = optim.Adam(D.parameters(), lr=lr)
#Q_optimizer = optim.Adam(Q.parameters(), lr=lr)
KL_params = list(Q.parameters()) + list(D.parameters())
KL_optimizer = optim.Adam(KL_params,lr)



# train loops

def D_train(x,y,z):
    
    D_loss = compute_D_loss(x,y,z)
    D_loss.backward()
    D_optimizer.step()
    
    return D_loss.data.item()

def G_train(x_u, y_u_pred, x_col, y_col_pred, z_u, z_col):
    G_loss, KL_loss, recon_loss, PDE_loss = compute_G_loss(x_u, y_u_pred, x_col, y_col_pred, z_u, z_col)
    G_loss.backward()
    KL_optimizer.step()
    
    return G_loss.data.item()
    
    

#%% Define inputs for trainig 

# xu len 1 
# x_col.shape = 100 collocation points 
# y_u = 1 point 


#print(data)



#%% 
#for epoch in range(1, n_epochs+1):           
D_losses, G_losses = [], []
   
z_u = torch.tensor(np.random.randn(z_dim)).float()

z_col = torch.tensor(np.random.randn(n_col,z_dim)).float()


    
idx = np.random.randint(0,n_col)

y_u_pred = D(torch.concat((x_u,z_u)))
#y_col_pred = compute_residuals(x_col[idx],z_col[idx])

u = G(torch.concat((x_col[idx],z_col[idx])))
print(u)
#u_t = torch.autograd.grad(
  #          u, t, 
   #         grad_outputs=torch.ones_like(u),
    #        retain_graph=True,
     #       create_graph=True
      #  )[0]

#u_tt = torch.autograd.grad(u_t.sum(), t, create_graph=True,allow_unused=True)[0]
#r_ode = m*u_tt+k*x
#print(r_ode) 




     
    #D_losses.append(D_train(x_u, y_u, z_u))
    #G_losses.append(G_train(x_u, y_u_pred, x_col, y_col_pred, z_u, z_col))

    #print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
     #       (epoch), n_epochs, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))
