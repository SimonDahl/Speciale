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
from scipy.integrate import odeint, solve_ivp



#%% Hyperparameters
n_sols = 2
bs = 1
n_neurons = 30
lr = 0.001
n_epochs = 100
z_dim = 100

time_limit = 5
gen_epochs = 5
timesteps = 100
SoftAdapt_start = 10000

t = np.linspace(0,time_limit,timesteps)

clip_value = 0.01
lambda_GP = 0

#np.random.seed(1234)

#%%

k = 5
def sho(t,y):
    m = np.random.uniform(1,10)
    c = np.random.uniform(1,5)
    solution = (y[1],(-(c/m)*y[1]-(k/m)*y[0]))
    return solution

train = np.zeros((n_sols,timesteps))

for i in range(n_sols):
    y_init = [np.random.uniform(1,10),np.random.uniform(1,10)]
    solution = solve_ivp(sho, [0,timesteps], y0 = y_init, t_eval = t)
    sol_data = solution.y[0]
    train[i,:] = sol_data
print('### DATA GENERATION COMPLETE ###')

#%%
t_plot = Variable(torch.from_numpy(t).float(), requires_grad=True).to(device)

train_set = torch.from_numpy(train)

train_set = train_set.float()
criterion = nn.BCELoss()
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=bs, shuffle=True)
#%% Generate data 



#%% Define Network



class Generator(nn.Module):
    def __init__(self, g_input_dim, g_output_dim):
        super(Generator, self).__init__()       
        self.fc1 = nn.Linear(g_input_dim, n_neurons)
        self.fc2 = nn.Linear(n_neurons, n_neurons)
        self.fc3 = nn.Linear(n_neurons, g_output_dim)
        
    def forward(self,y):
        y = torch.tanh(self.fc1(y))
        y = torch.tanh(self.fc2(y))
        return self.fc3(y)
    
class Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, n_neurons)
        self.fc2 = nn.Linear(n_neurons, 1)
    
    def forward(self, d):
        d = torch.tanh(self.fc1(d))
        return self.fc2(d)
    
G = Generator(g_input_dim = z_dim, g_output_dim = timesteps).to(device)
D = Discriminator(timesteps).to(device)

# set optimizer 
G_optimizer = optim.RMSprop(G.parameters(), lr=lr)
D_optimizer = optim.RMSprop(D.parameters(), lr=lr)


#%% Define training loops 
# 
def G_train(x):
    #=======================Train the generator=======================#
    G.zero_grad()

    z = Variable(torch.randn(bs, z_dim).to(device))
    y = Variable(torch.ones(bs, 1).to(device))

    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()
        
    return G_loss.data.item()

def D_train(x):
    #=======================Train the discriminator=======================#
    D.zero_grad()
    
    # train discriminator on real data
    
    x_real, y_real = x, torch.ones(bs,1)

    x_real, y_real = Variable(x_real.to(device)), Variable(y_real.to(device))

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # train discriminator on fake
    z = Variable(torch.randn(bs, z_dim).to(device))
    x_fake, y_fake = G(z), Variable(torch.zeros(bs, 1).to(device))
    

    D_output = D(x_fake)
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
    for batch_idx,x in enumerate(train_loader):
     
        D_losses.append(D_train(x))
        G_losses.append(G_train(x))

    print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
            (epoch), n_epochs, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))
                                                                                                                    
                                                                                                                    
with torch.no_grad():
    for i in range(5):
        z = Variable(torch.randn(z_dim).to(device))
        G_input = torch.cat((t_plot,z))
        generated = G(z)
        y = generated.cpu().detach().numpy()
        plt.plot(t,y)
   # plt.plot(t, train[0,:])
    #plt.savefig('./output/W_GAN/'+'n_epochs ' +str(n_epochs)+' z_dim_size '+str(z_dim)+' lr '+str(lr)+'.png')     
    plt.show()
    
                                                                                                                     