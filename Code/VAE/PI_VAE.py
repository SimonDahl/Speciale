from sqlite3 import Timestamp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.integrate import odeint
import argparse

from Code.PINN_HO import COL_RES, TRAIN_LIM

parser = argparse.ArgumentParser()

parser.add_argument('--n_epochs',help='Number of epochs',type=int)
parser.add_argument('--z_dim_size',help='Number of z dims',type=int)
parser.add_argument('--lr',help='Learning rate',type=float)


args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



#%%
# batch size 
bs = 5
#n_epochs = args.n_epochs
n_epochs = 1
# latent space size 
#z_dim_size = args.z_dim_size
z_dim_size = 3

#lr = args.lr   #3e-4
lr = 3e-4
timesteps = 500
n_data = 20


#print(test_set.shape)
#%%

class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2,h_dim3,h_dim4,z_dim):
        super(VAE, self).__init__()
        
       
        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc3 = nn.Linear(h_dim2, h_dim3)
        self.fc4 = nn.Linear(h_dim3, h_dim4)
        self.fc4_sigma = nn.Linear(h_dim4, z_dim)
        self.fc4_mu = nn.Linear(h_dim4, z_dim)
        # decoder part
        self.fc_z = nn.Linear(z_dim, h_dim4)
        self.fc5 = nn.Linear(h_dim4, h_dim3)
        self.fc6 = nn.Linear(h_dim3, h_dim2)
        self.fc7 = nn.Linear(h_dim2, h_dim1)
        self.fc8 = nn.Linear(h_dim1, x_dim)
        
        self.k = torch.nn.parameter.Parameter(torch.from_numpy(np.array([1])).float())
        self.m = torch.nn.parameter.Parameter(torch.from_numpy(np.array([1])).float())

        
    def encoder(self, x):
        h = torch.tanh(self.fc1(x))
        h = torch.tanh(self.fc2(h))
        h = torch.tanh(self.fc3(h))
        h = torch.tanh(self.fc4(h))
        return self.fc4_sigma(h), self.fc4_mu(h) # mu, log_var
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def decoder(self, z):
        h = torch.tanh(self.fc_z(z))
        h = torch.tanh(self.fc5(h))
        h = torch.tanh(self.fc6(h))
        h = torch.tanh(self.fc7(h))
        return (self.fc8(h)) 
    
    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, timesteps))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var
    
    
## PDE as loss function
def f(t,net):
    x = net(t)
    m,k  = net.getODEParam()
    x_t = torch.autograd.grad(x.sum(), t, create_graph=True)[0]
    x_tt = torch.autograd.grad(x_t.sum(), t, create_graph=True)[0]
    # Simlpe Harmonic Oscillator
    ode = m*x_tt+k*x
    return ode



h1= 50//2
h2 = h1//2
h3 = h2//2
h4 = h3//2

vae = VAE(x_dim=1, h_dim1=h1, h_dim2=h2,h_dim3=h3,h_dim4=h4, z_dim=z_dim_size)


if torch.cuda.is_available():
    vae.cuda()


#%%

optimizer = optim.Adam(vae.parameters(),lr=lr)


def loss_function(recon_x, x, mu, log_var,mse_u,mse_f,bp,cp,f,beta):
    MSE = F.mse_loss(recon_x, x.view(-1, timesteps), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    PI_loss = (beta*mse_u)/bp +(f*mse_f/cp)
    
    return MSE + KLD + PI_loss

  
#%%
# Generate Data for parameter estimation
TRAIN_LIM = 10
n = 1000
COL_RES = 1000

m_known = 1
k_known = 2
t_data = np.linspace(0,TRAIN_LIM,n)
y_data = np.cos(t_data*np.sqrt(k_known)) # Exact solution for (0,1) boundary condition



t_data = t_data.reshape(n,1)
t_data = Variable(torch.from_numpy(t_data).float(), requires_grad=True).to(device)
y_data = Variable(torch.from_numpy(y_data).float(), requires_grad=True).to(device)
y_data = y_data.reshape(n,1)
#Boundary Conditions
t_bc = np.array([[0]])
x_bc = np.array([[1]])

# Points and boundary vs ODE weight
col_points = int(TRAIN_LIM*COL_RES)
boundary_points = len(x_bc)+len(y_data)

F_WEIGHT = 1 #Physics Weight
B_WEIGHT = 1 #Boundary Weight
EPOCHS = 10
criterion = torch.nn.MSELoss() # Mean squared error for PI 




for epoch in range(EPOCHS):
    optimizer.zero_grad() # to make the gradients zero
    
    # Loss based on boundary conditions
    pt_t_bc = Variable(torch.from_numpy(t_bc).float(), requires_grad=True).to(device)
    pt_x_bc = Variable(torch.from_numpy(x_bc).float(), requires_grad=True).to(device)

    
    net_bc_out = vae(pt_t_bc) # output of u(x,t)

    net_data_out = vae(t_data)
    mse_u = criterion(input = net_bc_out, target = pt_x_bc)+criterion(input = net_data_out, target = y_data) # Boundary loss
    
    #mse_s = criterion(input = net_data_out, target = y_data) # Boundary loss
    


    # Loss based on PDE
    t_collocation = np.random.uniform(low=0.0, high=TRAIN_LIM, size=(col_points,1))
    all_zeros = np.zeros((col_points,1))    
    pt_t_collocation = Variable(torch.from_numpy(t_collocation).float(), requires_grad=True).to(device)
    pt_all_zeros = Variable(torch.from_numpy(all_zeros).float(), requires_grad=True).to(device)
    ode = f(pt_t_collocation,vae) # output of f(x,t)
    mse_f = criterion(input = ode, target = pt_all_zeros) #ODE Loss
    
    # Combining the loss functions
    L = loss_function(recon_x, x, mu, log_var,mse_u,mse_f,bp,cp,f,beta)
    #Gradients
    loss.backward() 
    #Step Optimizer
    optimizer.step() 
    #Display loss during training
    with torch.autograd.no_grad():
        if epoch%100== 0:
            print('Net Parameters:  k:',net.k.detach().numpy(),'m:',net.m.detach().numpy())
            print('Epoch:',epoch,"Traning Loss:",loss.data)
            print('Boundary Loss:',mse_u,'ODE Loss: ',100*mse_f)
        

