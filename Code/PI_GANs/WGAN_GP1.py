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

n_sols = 200
bs = 50
n_neurons = 30
lr = 0.001
n_epochs = 1000
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

train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=bs, shuffle=True)

#%%
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

#%%
G = Generator(g_input_dim = z_dim, g_output_dim = timesteps).to(device)
D = Discriminator(timesteps).to(device)

# set optimizer 
G_optimizer = optim.RMSprop(G.parameters(), lr=lr)
D_optimizer = optim.RMSprop(D.parameters(), lr=lr)

#%%
def wasserstein_loss_G(predict):
    loss = torch.mean(predict)
    return loss

def wasserstein_loss_D(true, predict):
    loss = torch.mean(true) - torch.mean(predict)
    return loss

def compute_gradient_penalty(y_real, y_pred):
    alpha = torch.rand(1)
    interpolates = (alpha * y_real + ((1 - alpha) * y_pred)).requires_grad_(True)
    d_interpolates = D(interpolates)

    fake = Variable(torch.Tensor(y_real.shape[0], 1).fill_(1.0), requires_grad=False)

    gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates, grad_outputs=fake, create_graph=True, retain_graph=True, only_inputs=True,)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def SoftAdapt(loss1,loss2):#, #mse_losses):
    eps = 10e-8
    n = 10
    s1 = np.zeros(n-1)
    s2 = np.zeros(n-1)
  
    
    l1 =  loss1[-n:]
    l2 =  loss2[-n:]
 
  
  
    for i in range(1,(n-1)):
        s1[i] = l1[i] - l1[i-1]
        s2[i] = l2[i] - l2[i-1]

    Beta = 0.1
    
    denominator = (np.exp(Beta*(s1[-1]-np.max(s1)))+np.exp(Beta*(s2[-1]-np.max(s2)))+eps)
    
    a1 =  (np.exp(Beta*(s1[-1]-np.max(s1))))/denominator
    a2 =  (np.exp(Beta*(s2[-1]-np.max(s2))))/denominator
   
    
    return a1,a2


#%% Define training loops 
# 
def G_train(z):
    #=======================Train the generator=======================#
    for g_epoch in range(gen_epochs):
        G_optimizer.zero_grad()

        y_pred= G(z)
        D_output = D(y_pred)
        
        G_loss = wasserstein_loss_G(D_output)
        G_loss.backward()
        G_optimizer.step()
        return G_loss



W_losses = []
GP_losses = []


def D_train(y_real, z,epoch):
    #=======================Train the discriminator=======================#
    D_optimizer.zero_grad()

    # train discriminator on real data
    D_real = D(y_real)

    # train discriminator on fake
    y_pred = G(z)
    D_fake = D(y_pred)
    
    W_loss = wasserstein_loss_D(D_real,D_fake)
    GP = compute_gradient_penalty(y_real,y_pred)
    
    D_loss = W_loss - GP
    
    
    if epoch > SoftAdapt_start-10:
        
        W_losses.append(W_loss)
        GP_losses.append(GP)
    
    a1,a2 =0,0 
    
    if epoch > SoftAdapt_start:
        a1,a2 = SoftAdapt(W_losses,GP_losses)
        D_loss = a1*W_loss - a2 * GP
        del W_losses[0]
        del GP_losses[0]
        

    
    D_loss.backward()
    D_optimizer.step()

    for p in D.parameters():
            p.data.clamp_(-clip_value, clip_value)
    return  D_loss,a1,a2

#%%
a1s,a2s = [],[]


for epoch in range(1, n_epochs+1):           
    D_losses, G_losses = [], []
    for batch_idx,data in enumerate(train_loader):
        y = data
        z = Variable(torch.randn(bs, z_dim).to(device))

        G_losses.append(G_train(z))
        D_loss,a1,a2 = D_train(y,z,epoch)
        D_losses.append(D_loss)
    a1s.append(a1)
    a2s.append(a2)

    print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
            (epoch), n_epochs, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))


""" 
e_plot = list(range(n_epochs))



plt.plot(e_plot,a1s)
plt.plot(e_plot,a2s)
plt.show() """

#print(a1s[-10:])
#print(a2s[-10:])
#%%
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
    
    