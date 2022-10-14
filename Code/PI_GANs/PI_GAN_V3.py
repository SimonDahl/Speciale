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


bs = 1
time_limit = 10
n_col = 100
m = 1
k = 2
x_data = np.linspace(0,time_limit,n_col)
y_data = np.cos(x_data*np.sqrt(k)) # Exact solution for (0,1) boundary condition
n_neurons = 50
lr = 0.001
drop = 0.0
z_dim = 1000
x_dim = x_data.shape[0]
y_dim = x_dim 
criterion = nn.BCELoss() 
criterion_ode = nn.MSELoss()
n_epochs = 100
gen_epoch = 5
#y_data = -k*np.cos()+k

#plt.plot(x_data,y_data)


x_data = x_data.reshape(n_col,1)
x_data = Variable(torch.from_numpy(x_data).float(), requires_grad=True).to(device)
y_data = Variable(torch.from_numpy(y_data).float(), requires_grad=True).to(device)
y_data = y_data.reshape(n_col,1)



class Generator(nn.Module):
    def __init__(self, g_input_dim, g_output_dim):
        super(Generator, self).__init__()       
        self.fc1 = nn.Linear(g_input_dim, n_neurons)
        self.fc2 = nn.Linear(n_neurons, n_neurons)
        self.fc3 = nn.Linear(n_neurons, n_neurons)
        self.fc4 = nn.Linear(n_neurons, n_neurons)
        self.fc5 = nn.Linear(n_neurons, g_output_dim)
    
        
        
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
        return (torch.sigmoid(self.fc5(d)))  # sigmoid for probaility 
        

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

# build network
G = Generator(g_input_dim = z_dim+x_dim, g_output_dim = y_dim).to(device)
D = Discriminator(x_dim+y_dim).to(device)
Q = Q_net(x_dim+y_dim,z_dim)

# set optimizer 
#G_optimizer = optim.Adam(G.parameters(), lr=lr)
D_optimizer = optim.Adam(D.parameters(), lr=lr)
#Q_optimizer = optim.Adam(Q.parameters(), lr=lr)
KL_params = list(Q.parameters()) + list(D.parameters())
KL_optimizer = optim.Adam(KL_params,lr)

# Physics-Informed residual on the collocation points         
def compute_residuals(x,z):
    g_input = torch.concat((x,z))
    u = G(g_input)
    u_t = torch.autograd.grad(u.sum(), x_data, create_graph=True)[0]
    u_tt = torch.autograd.grad(u_t.sum(), x_data, create_graph=True)[0]
    r_ode = m*u_tt+k*x
    return r_ode 


def G_train(x):
    
    #=======================Train the generator=======================#
    
    for g_epoch in range(gen_epoch):
    
        G.zero_grad()
        Q.zero_grad()
        z = Variable(torch.randn(z_dim,1).to(device))
        y_GAN = Variable(torch.ones(1).to(device))

        g_input = torch.concat((x,z))
        
        G_output = G(g_input[:,-1])
        D_output = D(torch.concat((x[:,-1],G_output)))

        G_loss = criterion(D_output, y_GAN)


        Q_input = torch.concat((x[:,-1],G_output))

        z_pred = Q(Q_input)

        ode_res = compute_residuals(x[:,-1],z_pred)

        ode_loss_target = torch.zeros_like(ode_res)
    
        ode_loss = torch.mean(criterion_ode(ode_res/n_col,ode_loss_target))

        loss = G_loss + ode_loss

        loss.backward()

        KL_optimizer.step()
       
    return loss.data.item()


def D_train(x):
    #=======================Train the discriminator=======================#
    D.zero_grad()
   
    # train discriminator on real data
    
    x_real, y_real = x, torch.ones(1)
    x_real, y_real = Variable(x_real.to(device)), Variable(y_real.to(device))

    D_output = D(torch.concat((x[:,-1],y_data[:,-1])))
    D_real_loss = criterion(D_output, y_real)

    D_real_score = D_output

    # train discriminator on fake
    z = Variable(torch.randn(z_dim,1).to(device))
    g_input = torch.concat((x,z))

    x_fake, y_fake = G(g_input[:,-1]), Variable(torch.zeros(1).to(device))
    
    D_output = D(torch.concat((x[:,-1],x_fake)))
    D_fake_loss = criterion(D_output, y_fake)

    D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss

    D_loss.backward()
    D_optimizer.step()

        
    return  D_loss.data.item()


#%% 
for epoch in range(1, n_epochs+1):           
    D_losses, G_losses = [], []
    D_losses.append(D_train(x_data))
    G_losses.append(G_train(x_data))

    print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
            (epoch), n_epochs, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))
                                                                                                                    
                                                                                                                    
#%% Generate sample 


with torch.no_grad():
    
    z = Variable(torch.randn(z_dim).to(device))
    g_gen = torch.concat((x_data[:,-1],z))
    generated = G(g_gen)
    y = generated.cpu().detach().numpy()
    plt.plot(x_data[:,-1],y)
    plt.show()


