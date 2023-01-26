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
import random
import argparse
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from scipy.integrate import odeint, solve_ivp

parser = argparse.ArgumentParser()

parser.add_argument('--n_epochs',help='Number of epochs',type=int)
parser.add_argument('--n_sols',help='Number of solutions',type=int)
parser.add_argument('--w_gan_gp',help='Type of GAN',type=int)
parser.add_argument('--opt_choice',help='RMSprop or ADAM',type=int)

args = parser.parse_args()
#%%

HPC = False
if HPC:
    n_sols = args.n_sols
    n_epochs = args.n_epochs
    w_gan_gp = args.w_gan_gp
    opt_choice = args.opt_choice
else:
    n_sols = 1000
    n_epochs = 100
    w_gan_gp = 2 # 0 - GAN, 1 - W_GAN, 2 - W_GAN_GP
    opt_choice = 1 # 0 - RMSprop, 1 - ADAM 

bs = n_sols // 10
lr = 0.0001
criterion = nn.BCELoss() 
z_dim = 100

time_limit = 10
critic_epochs = 5
timesteps = 250
t = np.linspace(0,time_limit,timesteps)

slope = 0.01
drop = 0.2
clip_value = 0.01
lambda_GP = 10

np.random.seed(12345)

#%%

def sho(t,y):
    m = np.random.uniform(0.1, 2)
    k = np.random.uniform(3, 10)
    c = np.random.uniform(0, 1)
    solution = (y[1],(-(c/m)*y[1]-(k/m)*y[0]))
    return solution

train = np.zeros((n_sols,timesteps))

for i in range(n_sols):
    y_init = [np.random.uniform(1,3),np.random.uniform(0,1)]
    solution = solve_ivp(sho, [0,time_limit], y0 = y_init, t_eval = t)
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
        self.fc1 = nn.Linear(g_input_dim, g_input_dim*2)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features*2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features*2)
        self.fc4 = nn.Linear(self.fc3.out_features, self.fc3.out_features*2)
        self.fc5 = nn.Linear(self.fc4.out_features, g_output_dim)
    
    # forward method
    def forward(self, x): 
        x = F.leaky_relu(self.fc1(x), slope)
        x = F.leaky_relu(self.fc2(x), slope) 
        x = F.leaky_relu(self.fc3(x), slope)
        x = F.leaky_relu(self.fc4(x), slope)
        return self.fc5(x) 
    
class Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, d_input_dim *4)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
        self.fc4 = nn.Linear(self.fc3.out_features, self.fc3.out_features//2)
        self.fc5 = nn.Linear(self.fc4.out_features, 1)  # output dim = 1 for binary classification 
    
    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), slope)
        x = F.dropout(x, drop)
        x = F.leaky_relu(self.fc2(x), slope)
        x = F.dropout(x, drop)
        x = F.leaky_relu(self.fc3(x), slope)
        x = F.dropout(x, drop)
        x = F.leaky_relu(self.fc4(x), slope)
        x = F.dropout(x, drop)
        if w_gan_gp == 0:
            return torch.sigmoid(self.fc5(x))
        return self.fc5(x)

#%%
G = Generator(g_input_dim = z_dim, g_output_dim = timesteps).to(device)
D = Discriminator(timesteps).to(device)

# set optimizer
if opt_choice == 0:
    G_optimizer = optim.RMSprop(G.parameters(), lr=lr)
    D_optimizer = optim.RMSprop(D.parameters(), lr=lr)
else: 
    G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0, 0.9))
    D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0, 0.9))

#%%
def wasserstein_loss_G(predict):
    loss = -torch.mean(predict)
    return loss

def wasserstein_loss_D(true, predict):
    loss = -(torch.mean(true) - torch.mean(predict))
    return loss

def compute_gradient_penalty(y_real, y_pred):
    eps = Variable(torch.rand(bs, 1).to(device))
    interpolates = (eps * y_real + (1 - eps) * y_pred).requires_grad_(True)
    d_interpolates = D(interpolates)

    #fake = Variable(torch.Tensor(y_real.shape[0], 1).fill_(1.0), requires_grad=False).to(device)
    fake = torch.ones_like(d_interpolates)

    gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates, grad_outputs=fake, create_graph=True, retain_graph=True,)[0]
    #gradients = gradients.view(gradients.size(0), -1)
    gradients = gradients.view(bs, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

#%% Define training loops 
# 
def G_train():
    #=======================Train the generator=======================#
    G_optimizer.zero_grad()

    z = Variable(torch.randn(bs, z_dim).to(device))
    y_pred= G(z)
    D_output = D(y_pred)
    if w_gan_gp == 0:
        y = Variable(torch.ones(bs, 1).to(device))
        G_loss = criterion(D_output, y)
    else:
        G_loss = wasserstein_loss_G(D_output)
    G_loss.backward()
    G_optimizer.step()
    return G_loss

def D_train(y_real):
    #=======================Train the discriminator=======================#
    for c_epoch in range(critic_epochs):
        D_optimizer.zero_grad()

        y_real = y_real.to(device)
        # train discriminator on real data
        D_real = D(y_real)

        # train discriminator on fake
        z = Variable(torch.randn(bs, z_dim).to(device))
        y_pred = G(z)
        D_fake = D(y_pred)

        if w_gan_gp == 0:
            y_real = Variable(torch.ones(bs, 1).to(device))
            y_pred = Variable(torch.zeros(bs, 1).to(device))
            D_real = criterion(D_real, y_real)
            D_fake = criterion(D_fake, y_pred)
            D_loss = D_real + D_fake
        elif w_gan_gp == 1:
            D_loss = wasserstein_loss_D(D_real,D_fake)
        else:
            D_loss = torch.mean(D_fake) - torch.mean(D_real)
            GP = compute_gradient_penalty(y_real,y_pred)
            D_loss = D_loss + lambda_GP * GP
        D_loss.backward()
        D_optimizer.step()
        if w_gan_gp == 1:
            for p in D.parameters():
                p.data.clamp_(-clip_value, clip_value)
    return  D_loss

#%%
start = time.time()

for epoch in range(1, n_epochs+1):           
    D_losses, G_losses = [], []
    for batch_idx,data in enumerate(train_loader):
        y = data
        D_losses.append(D_train(y))
        G_losses.append(G_train())

    print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
            (epoch), n_epochs, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))

end = time.time() 
print("Time elapsed during the calculation:", end - start) 

#%%
#%% Generate sample 

with torch.no_grad():
    for i in range(5):
        z = Variable(torch.randn(z_dim).to(device))
        generated = G(z)
        y = generated.cpu().detach().numpy()
        plt.plot(t,y)
        plt.xlabel('Time')
        plt.ylabel('Position')
    if HPC:
        plt.savefig('./output/W_GAN/Pendulum/'+' n_epochs ' +str(n_epochs)+' n_sols '+str(n_sols)+' w_gan_gp'+str(w_gan_gp)+' opt_choice'+str(opt_choice)+'.png')     
    else:
        plt.show()

with torch.no_grad():
    fig, ax = plt.subplots(2,4)
    fig.set_figheight(10)
    fig.set_figwidth(20)
    c = 1
    for i in range(0,2):
        for j in range(0,4):
            z = Variable(torch.randn(z_dim).to(device))
            generated = G(z)
            y = generated.cpu().detach().numpy()
            ax[i,j].plot(t,y)
            c+= 1
    ax[0,0].set_xlabel('Time')
    ax[0,0].set_ylabel('Position')
    ax[1,0].set_xlabel('Time')
    ax[1,0].set_ylabel('Position')
    if HPC:
        plt.savefig('./output/W_GAN/Pendulum/'+'Sub-plot'+' n_epochs ' +str(n_epochs)+' n_sols '+str(n_sols)+' w_gan_gp'+str(w_gan_gp)+' opt_choice'+str(opt_choice)+'.png')  
    else:
        plt.show()  
    