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

parser = argparse.ArgumentParser()

parser.add_argument('--n_epochs',help='Number of epochs',type=int)
parser.add_argument('--n_sols',help='Number of solutions',type=int)
parser.add_argument('--w_gan_gp',help='Type of GAN',type=int)

args = parser.parse_args()
#%%


HPC = False

if HPC:
    n_sols = args.n_sols
    n_epochs = args.n_epochs
    w_gan_gp = args.w_gan_gp
else:
    n_sols = 30
    n_epochs = 1
    w_gan_gp = 1
    

bs = n_sols // 10
n_neurons = 50
lr = 0.001
criterion = nn.BCELoss() 

#
z_dim = 100

time_limit = 5
gen_epochs = 5
timesteps = 100
t = np.linspace(0,time_limit,timesteps)

clip_value = 0.01
lambda_GP = 1
# # 0 - GAN, 1 - W_GAN, 2 - W_GAN_GP


np.random.seed(12345)


m = 2
k = 5
c = 1

#%%

def sho(t,y):
  #  m = np.random.uniform(0.1,2)
   # k = np.random.uniform(3,10)
    #c = np.random.uniform(0,1)
    solution = (y[1],(-(c/m)*y[1]-(k/m)*y[0]))
    return solution

train = np.zeros((n_sols,timesteps))


print('### DATA GENERATION Started ###')
for i in range(n_sols):
    y_init = [np.random.uniform(1,5),np.random.uniform(-2,2)]
    solution = solve_ivp(sho, [0,timesteps], y0 = y_init, t_eval = t)
    sol_data = solution.y[0]
    train[i,:] = sol_data
print('### DATA GENERATION COMPLETE ###')



for i in range(n_sols):
    plt.plot(t,train[i,:])
plt.title('Range of solutions')
plt.xlabel('Time')
plt.ylabel('Postion')
plt.show()


#%%
t_plot = Variable(torch.from_numpy(t).float(), requires_grad=True).to(device)

train_set = torch.from_numpy(train)

train_set = train_set.float()

train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=bs, shuffle=True)
""" 
#%%
class Generator(nn.Module):
    def __init__(self, g_input_dim, g_output_dim):
        super(Generator, self).__init__()       
        self.fc1 = nn.Linear(g_input_dim, n_neurons)
        self.fc2 = nn.Linear(n_neurons, n_neurons)
        self.fc3 = nn.Linear(n_neurons, n_neurons)
        self.fc4 = nn.Linear(n_neurons, n_neurons)
        self.fc5 = nn.Linear(n_neurons, g_output_dim)
        
    def forward(self,y):
        y = torch.tanh(self.fc1(y))
        y = torch.tanh(self.fc2(y))
        y = torch.tanh(self.fc3(y))
        y = torch.tanh(self.fc4(y))
        return self.fc5(y)
    
class Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, n_neurons)
        self.fc2 = nn.Linear(n_neurons, n_neurons)
        self.fc3 = nn.Linear(n_neurons, n_neurons)
        self.fc4 = nn.Linear(n_neurons, 1)
    
    def forward(self, d):
        d = torch.tanh(self.fc1(d))
        d = torch.tanh(self.fc2(d))
        d = torch.tanh(self.fc3(d))
        if (w_gan_gp == 0):
            return torch.sigmoid(self.fc4(d))
        return self.fc4(d)

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

#%% Define training loops 
# 
def G_train(z):
    #=======================Train the generator=======================#
    for g_epoch in range(gen_epochs):
        G_optimizer.zero_grad()

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

def D_train(y_real, z):
    #=======================Train the discriminator=======================#
    D_optimizer.zero_grad()
    y_real = Variable(y_real.to(device))
    # train discriminator on real data
    D_real = D(y_real)

    # train discriminator on fake
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
        D_loss = wasserstein_loss_D(D_real,D_fake)
        GP = compute_gradient_penalty(y_real,y_pred)
        D_loss = D_loss - lambda_GP * GP
    D_loss.backward()
    D_optimizer.step()

    for p in D.parameters():
            p.data.clamp_(-clip_value, clip_value)
    return  D_loss

#%%
for epoch in range(1, n_epochs+1):           
    D_losses, G_losses = [], []
    for batch_idx,data in enumerate(train_loader):
        y = data
        z = Variable(torch.randn(bs, z_dim).to(device))

        D_losses.append(D_train(y, z))
        G_losses.append(G_train(z))

    print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
            (epoch), n_epochs, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))

#%%
with torch.no_grad():
    for i in range(5):
        z = Variable(torch.randn(z_dim).to(device))
        G_input = torch.cat((t_plot,z))
        generated = G(z)
        y = generated.cpu().detach().numpy()
        plt.plot(t,y)
    #plt.savefig('/Users/andreasgleerup/Desktop/Figures/W_GAN/'+'W_GAN_GP '+str(w_gan_gp)+' n_epochs ' +str(n_epochs)+' lr '+str(lr)+'.png')     
    plt.savefig('./output/W_GAN/'+'W_GAN_GP '+str(w_gan_gp)+' n_epochs ' +str(n_epochs)+' n_sols '+str(n_sols)+'.png')     
    #plt.show()

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
            ax[i,j].set_title('Sample ' + str(c))
            c+= 1
    plt.savefig('./output/W_GAN/'+'Sub-plot W_GAN_GP '+str(w_gan_gp)+' n_epochs ' +str(n_epochs)+' n_sols '+str(n_sols)+'.png')
    #plt.show() """