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
from scipy.integrate import odeint, solve_ivp
import argparse


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#%%
# batch size 
bs = 1
n_epochs = 100
z_dim_size = 3
phy = 1
np.random.seed(1234)
lr = 0.001
timesteps = 400
n_data = 100
time_limit = 3
n_col = 100



#%% Generate data 

t = np.linspace(0,time_limit,timesteps)
sol_data = np.zeros((n_data,timesteps))

#y = [2,1]
m = 1
k = 2
c = 3
def sho(t,y):
    solution = (y[1],(-(c/m)*y[1]-(k/m)*y[0])) 
    return solution

for i in range(n_data):
    y_init = [np.random.uniform(0,5),0]
    solution = solve_ivp(sho, [0,timesteps], y0 = y_init, t_eval = t)
    sol_data[i,:] = solution.y[0]



   

x_col =  np.linspace(0, time_limit, n_col)
t_diff = Variable(torch.from_numpy(t).float(), requires_grad=True).to(device) 
t_diff = t_diff.reshape(timesteps,1)
#x_col = x_col.reshape(n_col,1)
x_col = Variable(torch.from_numpy(x_col).float(), requires_grad=True).to(device)




#%%

# split into test, validation, and training sets
y_temp, y_test, _, _ = train_test_split(sol_data, sol_data, test_size=0.05)
y_train, y_valid, _, _ = train_test_split(y_temp,
                                          y_temp,
                                          test_size=0.1)
n_train = len(y_train)
n_valid = len(y_valid)
n_test = len(y_test)

#print(n_train)


train_set = torch.from_numpy(y_train)
test_set = torch.from_numpy(y_test)

train_set = train_set.float()
test_set = test_set.float()

train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=bs, shuffle=False)




class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2,h_dim3,h_dim4,z_dim):
        super(VAE, self).__init__()
        
       
        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc3 = nn.Linear(h_dim2, h_dim3)
        self.fc4_sigma = nn.Linear(h_dim3, z_dim)
        self.fc4_mu = nn.Linear(h_dim3, z_dim)
        # decoder part
        self.fc_z = nn.Linear(z_dim, h_dim3)
        self.fc5 = nn.Linear(h_dim3, h_dim2)
        self.fc6 = nn.Linear(h_dim2, h_dim1)
        self.fc7 = nn.Linear(h_dim1, x_dim)
        
    def encoder(self, x):
        h = F.leaky_relu(self.fc1(x))
        h = F.leaky_relu(self.fc2(h))
        h = F.leaky_relu(self.fc3(h))
        
        return self.fc4_sigma(h), self.fc4_mu(h) # mu, log_var
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def decoder(self, z):
        h = F.leaky_relu(self.fc_z(z))
        h = F.leaky_relu(self.fc5(h))
        h = F.leaky_relu(self.fc6(h))
        return (self.fc7(h)) 
    
    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, timesteps))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var

h1= timesteps//2
h2 = h1//2
h3 = h2//2
h4 = h3//2

vae = VAE(x_dim=timesteps, h_dim1=h1, h_dim2=h2,h_dim3=h3,h_dim4=h4, z_dim=z_dim_size)


def compute_residuals(u):
    
    u = u.reshape(timesteps,1)
    plt.plot(t_diff.detach().numpy(),u.detach().numpy())
    plt.show()
  
    #u_t  = torch.autograd.grad(u, t_diff, torch.ones_like(u), retain_graph=True,create_graph=True)[0]# computes du/dx
    u_t = torch.autograd.grad(u.sum(), t_diff, create_graph=True)[0]
    u_tt = torch.autograd.grad(u_t.sum(), t_diff, create_graph=True)[0]
    #u_tt = torch.autograd.grad(u_t,  t_diff, torch.ones_like(u_t),retain_graph=True ,create_graph=True)[0]# computes d^2u/dx^2

    
    r_ode = m*u_tt+c*u_t + k*u # damped harmonic oscillator 
    
     
    return r_ode


optimizer = optim.Adam(vae.parameters(),lr=lr)

def loss_function(recon_x, x, mu, log_var,x_col):
    MSE = F.mse_loss(recon_x, x.view(-1, timesteps), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    r_ode = compute_residuals(recon_x)
    return MSE + KLD +  r_ode 


def train(epoch):
    #vae.train()
    train_loss = 0
    for batch_idx, (data) in enumerate(train_loader):
        if torch.cuda.is_available():
            data = data.cuda()
        optimizer.zero_grad()
        
        recon_batch, mu, log_var = vae(data)
        
        loss = loss_function(recon_batch, data, mu, log_var,x_col)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        
    if epoch % 20 == 0:
      print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))    
    


for epoch in range(1, n_epochs):
    train(epoch)
   

 
 