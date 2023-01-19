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
import scipy
from scipy.interpolate import griddata
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.nn.utils import weight_norm as wn

#%% Hyperparameters 

np.random.seed(1234)
n_neurons = 40
n_neurons_d = 128
lr = 0.001
criterion_MSE = nn.MSELoss() # loss function
criterion_BCE = nn.BCELoss() 
lambda_phy = 0.1
D_epochs = 1
p_save = 1 # save lambda values at every epoch % p_save interval 


#%% HPC and data load 

HPC = False 

if HPC == True:
    print('Started code')
    n_epochs = 100000
    N_train = 1000
    data = scipy.io.loadmat('cylinder_nektar_wake.mat')
if HPC == False: 
    n_epochs = 100
    N_train = 50
    
    data = scipy.io.loadmat(r"C:\Users\Simon\OneDrive - Danmarks Tekniske Universitet\Speciale\Speciale\Code\NSGAN\cylinder_nektar_wake.mat")

bs = N_train//10 
lambda_1_true = 1
lambda_2_true = 0.01


#%% Ready data 

U_star = data['U_star'] # N x 2 x T
P_star = data['p_star'] # N x T
t_star = data['t'] # T x 1
X_star = data['X_star'] # N x 2

N = X_star.shape[0]
T = t_star.shape[0]

# Rearrange Data 
XX = np.tile(X_star[:,0:1], (1,T)) # N x T
YY = np.tile(X_star[:,1:2], (1,T)) # N x T
TT = np.tile(t_star, (1,N)).T # N x T

UU = U_star[:,0,:] # N x T
VV = U_star[:,1,:] # N x T
PP = P_star # N x T

x = XX.flatten()[:,None] # NT x 1
y = YY.flatten()[:,None] # NT x 1
t = TT.flatten()[:,None] # NT x 1

u = UU.flatten()[:,None] # NT x 1
v = VV.flatten()[:,None] # NT x 1
p = PP.flatten()[:,None] # NT x 1

 # Training Data    
idx = np.random.choice(N*T, N_train, replace=False)
x_train = x[idx,:]
y_train = y[idx,:]
t_train = t[idx,:]
u_train = u[idx,:]
v_train = v[idx,:]
p_train = p[idx,:]



x_train = Variable(torch.from_numpy(x_train).float(), requires_grad=True).to(device)
y_train = Variable(torch.from_numpy(y_train).float(), requires_grad=True).to(device)
t_train = Variable(torch.from_numpy(t_train).float(), requires_grad=True).to(device)
u_train = Variable(torch.from_numpy(u_train).float(), requires_grad=True).to(device)
v_train = Variable(torch.from_numpy(v_train).float(), requires_grad=True).to(device)
p_train = Variable(torch.from_numpy(p_train).float(), requires_grad=True).to(device)


#%% Define networks 

class Generator(nn.Module):
    def __init__(self, g_input_dim, g_output_dim):
        super(Generator, self).__init__()       
        self.fc1 = wn(nn.Linear(g_input_dim, n_neurons)) # apply weight normalization to all layers 

        self.fc2 = wn(nn.Linear(n_neurons, n_neurons))
        self.fc3 = wn(nn.Linear(n_neurons, n_neurons))
        self.fc4 = wn(nn.Linear(n_neurons, n_neurons))
        self.fc5 = wn(nn.Linear(n_neurons, n_neurons))
        self.fc6 = wn(nn.Linear(n_neurons, n_neurons))
        self.fc7 = wn(nn.Linear(n_neurons, n_neurons))
        self.fc8 = wn(nn.Linear(n_neurons, n_neurons))
        self.fc9 = wn(nn.Linear(n_neurons, n_neurons))
        self.fc10 = wn(nn.Linear(n_neurons, n_neurons))
        self.fc11 = wn(nn.Linear(n_neurons, n_neurons))
        self.fc12 = wn(nn.Linear(n_neurons, g_output_dim))

        self.lambda1 = torch.nn.parameter.Parameter(torch.from_numpy(np.array([0.5])).float(), requires_grad=True)
        self.lambda2 = torch.nn.parameter.Parameter(torch.from_numpy(np.array([0.5])).float(), requires_grad=True)
                
      
    # forward method
    def forward(self,y):
        y = F.silu(self.fc1(y)) 
        y = F.silu(self.fc2(y)) 
        y = F.silu(self.fc3(y))
        y = F.silu(self.fc4(y))
        y = F.silu(self.fc5(y))
        y = F.silu(self.fc6(y))
        y = F.silu(self.fc7(y))
        y = F.silu(self.fc8(y))
        y = F.silu(self.fc9(y))
        y = F.silu(self.fc10(y))
        y = F.silu(self.fc11(y))
        return self.fc12(y) 

    
    def getPDEParam(self):
        return (self.lambda1,self.lambda2)

class Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = wn(nn.Conv2d(1, 4, 3, stride=1))
        self.fc2 = wn(nn.Conv2d(4, 8, 1, stride=1))
        self.fc3 = wn(nn.Conv2d(8, 16, 1, stride=1))
        self.fc4 = wn(nn.Linear(1,n_neurons_d))
        self.fc5 = wn(nn.Linear(n_neurons_d,n_neurons_d))
        self.fc6 = wn(nn.Linear(n_neurons_d, 1))  # output dim = 1 for binary classification 
    
    # forward method
    def forward(self, d):
        d = torch.relu(self.fc1(d))
        d = torch.relu(self.fc2(d))
        d = torch.relu(self.fc3(d))
        d = torch.relu(self.fc4(d))
        d = torch.relu(self.fc5(d))
        return torch.sigmoid(self.fc6(d))

#%%  build networks

G = Generator(g_input_dim =3, g_output_dim = 2).to(device)
D = Discriminator(d_input_dim = 2).to(device)
# set optimizer 
G_optimizer = optim.Adam(G.parameters(), lr=lr)
D_optimizer = optim.Adam(D.parameters(), lr=lr)


#%% define functions 

def predict(x,y,t):

    psi_and_p = G(torch.concat((x,y,t),dim=1))
    psi = psi_and_p[:,0:1]
    p = psi_and_p[:,1:2]
    
    u = torch.autograd.grad(psi, y, torch.ones_like(y), retain_graph=True,create_graph=True)[0]# 
    v = -torch.autograd.grad(psi, x, torch.ones_like(x), retain_graph=True,create_graph=True)[0]#
    lambda1,lambda2 = G.getPDEParam()
    return p,u,v,lambda1,lambda2

def compute_residuals(x, y, t):

    
    psi_and_p = G(torch.concat((x,y,t),dim=1))
    psi = psi_and_p[:,0:1]
    p = psi_and_p[:,1:2]


    u = torch.autograd.grad(psi, y, torch.ones_like(y), retain_graph=True,create_graph=True)[0]# 
    v = -torch.autograd.grad(psi, x, torch.ones_like(x), retain_graph=True,create_graph=True)[0]#
    
    u_t = torch.autograd.grad(u, t, torch.ones_like(t), retain_graph=True,create_graph=True)[0]# 
    u_x = torch.autograd.grad(u, x, torch.ones_like(x), retain_graph=True,create_graph=True)[0]# 
    u_y = torch.autograd.grad(u, y, torch.ones_like(y), retain_graph=True,create_graph=True)[0]# 

    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), retain_graph=True,create_graph=True)[0]# 
    u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), retain_graph=True,create_graph=True)[0]# 

    v_t = torch.autograd.grad(v, t, torch.ones_like(t), retain_graph=True,create_graph=True)[0]# 
    v_x = torch.autograd.grad(v, x, torch.ones_like(x), retain_graph=True,create_graph=True)[0]# 
    v_y = torch.autograd.grad(u, y, torch.ones_like(y), retain_graph=True,create_graph=True)[0]# 
   
    v_xx = torch.autograd.grad(v_x, x, torch.ones_like(x), retain_graph=True,create_graph=True)[0]# 
    v_yy = torch.autograd.grad(v_y, y, torch.ones_like(y), retain_graph=True,create_graph=True)[0]#  

    p_x = torch.autograd.grad(p, x, torch.ones_like(x), retain_graph=True,create_graph=True)[0]# 
    p_y = torch.autograd.grad(p, y, torch.ones_like(y), retain_graph=True,create_graph=True)[0]# 

    lambda_1, lambda_2 = G.getPDEParam()
  
    f_u = u_t + lambda_1*(u*u_x + v*u_y) + p_x - lambda_2*(u_xx + u_yy) 
    f_v = v_t + lambda_1*(u*v_x + v*v_y) + p_y - lambda_2*(v_xx + v_yy)
    
    return u, v, p, f_u, f_v


def rearange(x,y,t,p,u,v):
    img = torch.zeros(bs,1,3,3)
    for i in range(bs):
        row1 = torch.tensor([x[i],y[i],t[i]])
        row2 = torch.tensor([p[i],u[i],v[i]])
        img[i,0,0,:] = row1
        img[i,0,1,:] = row2
        img[i,0,2,:] = row1
              
    return Variable(img).to(device)


#%% Generator traning loop 

def G_train(x,y,t,p,u,v):
    
    G.zero_grad()
    
    # MSE loss on training points
    p_fake,u_fake,v_fake,_,_ = predict(x,y,t)
    
    MSE_p = criterion_MSE(p,p_fake)
    MSE_u = criterion_MSE(u,u_fake)
    MSE_v = criterion_MSE(v,v_fake)
      
    _, _, _, f_u, f_v = compute_residuals(x,y,t)
    target = torch.zeros_like(f_u)

    MSE_f_u = criterion_MSE(f_u,target)
    MSE_f_v = criterion_MSE(f_v,target)
    L_phy = MSE_f_u + MSE_f_v
      
    d_input = rearange(x,y,t,p_fake,u_fake,v_fake)
    L_adv =torch.mean(-torch.log(D(d_input)+1e-8)) # Advisarial loss 
           
    G_loss = L_adv + lambda_phy * L_phy + MSE_p + MSE_u + MSE_v
    
    G_loss.backward()
    G_optimizer.step()
    MSE_loss = MSE_p + MSE_u + MSE_v

    return G_loss, MSE_loss 

#%% Discriminator tranining loop 

def D_train(x,y,t,p,u,v):
    
    for d_epoch in range(D_epochs):
        D.zero_grad()
        
        p_fake,u_fake,v_fake,_,_ = predict(x,y,t)
        d_input = rearange(x,y,t,p_fake,u_fake,v_fake)    
        fake_prob = D(d_input)
        

        d_input = rearange(x,y,t,p,u,v)
        real_prob = D(d_input)
        
        D_loss = torch.mean(-torch.log(real_prob+1e-8)+torch.log(1-fake_prob+1e-8))
                
        D_loss.backward()
        D_optimizer.step()
    
    return D_loss

#%%& Training loop 


lambda1_list = []
lambda2_list = []


for epoch in range(1, n_epochs+1):
    D_losses, G_losses, MSE_losses = [], [], []

    for batch_idx in range(N_train//bs):
        idx = np.random.choice(N_train, bs, replace=False)
        x_batch = x_train[idx,:]
        y_batch = x_train[idx,:]
        t_batch = x_train[idx,:]
        u_batch = x_train[idx,:]
        v_batch = x_train[idx,:]
        p_batch = x_train[idx,:]
        G_loss,MSE_loss =  G_train(x_batch,y_batch,t_batch,p_batch,u_batch,v_batch)
        G_losses.append(G_loss)
        MSE_losses.append(MSE_loss)
        D_losses.append(D_train(x_batch,y_batch,t_batch,p_batch,u_batch,v_batch))
    
    if epoch % p_save == 0: 
        _,_, _,lambda1_approx,lambda2_approx = predict(x_train, y_train, t_train)
        lambda1_list.append(lambda1_approx)
        lambda2_list.append(lambda2_approx)

    print('[%d/%d]: loss_d: %.3f, loss_g: %.3f, loss_MSE: %.3f' % (
            (epoch), n_epochs, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses)),torch.mean(torch.FloatTensor(MSE_losses))))


#%% Test  

# Test Data
snap = np.array([100])
x_star = X_star[:,0:1]
y_star = X_star[:,1:2]
t_star = TT[:,snap]

u_star = U_star[:,0,snap]
v_star = U_star[:,1,snap]
p_star = P_star[:,snap]

x_star = Variable(torch.from_numpy(x_star).float(), requires_grad=True).to(device)
y_star = Variable(torch.from_numpy(y_star).float(), requires_grad=True).to(device)
t_star = Variable(torch.from_numpy(t_star).float(), requires_grad=True).to(device)
u_star = Variable(torch.from_numpy(u_star).float(), requires_grad=True).to(device)
v_star = Variable(torch.from_numpy(v_star).float(), requires_grad=True).to(device)
p_star = Variable(torch.from_numpy(p_star).float(), requires_grad=True).to(device)

p_pred,u_pred, v_pred,lambda1_approx,lambda2_approx = predict(x_star, y_star, t_star)

# Error u,v,p 


def plot_solution(X_star, u_star, index):
    
    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = 200
    x = np.linspace(lb[0], ub[0], nn)
    y = np.linspace(lb[1], ub[1], nn)
    X, Y = np.meshgrid(x,y)
    
    U_star = griddata(X_star, u_star.flatten(), (X, Y), method='cubic')
    
    plt.figure(index)
    plt.pcolor(X,Y,U_star, cmap = 'jet')
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    if index == 3:
        plt.title('Predicted pressure')
    if index == 4:
        plt.title('Exact pressure')
    if index == 5:
        plt.title('Exact-predicted pressure')

    if HPC == False: 
        plt.show()
    if HPC == True: 
        plt.savefig('./output/NSGAN/'+'Plot Discovery ' +str(index)+'.png')
         

#%% Plot and print results 

with torch.no_grad():

    u_star = u_star.cpu().detach().numpy()
    u_pred = u_pred.cpu().detach().numpy()
    v_star = v_star.cpu().detach().numpy()
    v_pred = v_pred.cpu().detach().numpy()
    p_star = p_star.cpu().detach().numpy()
    p_pred = p_pred.cpu().detach().numpy()

    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    error_v = np.linalg.norm(v_star-v_pred,2)/np.linalg.norm(v_star,2)
    error_p = np.linalg.norm(p_star-p_pred,2)/np.linalg.norm(p_star,2)
    error_lambda1 =np.abs(lambda_1_true-lambda1_approx)
    error_lambda2 = np.abs(lambda_2_true-lambda2_approx)
        
    print('Error u: %e' % (error_u))    
    print('Error v: %e' % (error_v))    
    print('Error p: %e' % (error_p)) 
    print('Error lambda_1: %e' % (error_lambda1))
    print('Error lambda_2: %e' % (error_lambda2))
    print('lambda_1 Approx: %e' % (lambda1_approx))
    print('lambda_2 Approx: %e' % (lambda2_approx))     

     # Predict for plotting
    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = 200
    x = np.linspace(lb[0], ub[0], nn)
    y = np.linspace(lb[1], ub[1], nn)
    X, Y = np.meshgrid(x,y)
    
    UU_star = griddata(X_star, u_pred.flatten(), (X, Y), method='cubic')
    VV_star = griddata(X_star, v_pred.flatten(), (X, Y), method='cubic')
    PP_star = griddata(X_star, p_pred.flatten(), (X, Y), method='cubic')
    P_exact = griddata(X_star, p_star.flatten(), (X, Y), method='cubic')

   

    plot_solution(X_star, p_pred, 3)    
    plot_solution(X_star, p_star, 4)
    plot_solution(X_star, p_star - p_pred, 5)
    
    e_plot = list(range(len(lambda1_list)))
    lambda1_true_list = np.repeat(lambda_1_true,len(lambda1_list))
    lambda2_true_list = np.repeat(lambda_2_true,len(lambda1_list))

    plt.plot(e_plot,lambda1_true_list,label='lambda_1 True',color='red')
    plt.plot(e_plot,lambda1_list,'--',label='lambda_1 approx',color='red')
    plt.plot(e_plot,lambda2_true_list,label='lambda_2 True',color='blue')
    plt.plot(e_plot,lambda2_list,'--',label='lambda_2 approx',color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend()
    plt.title('Data driven discovery of parameters lambda_1 and lambda_2')
    
    if HPC == True: 
        plt.savefig('./output/NSGAN/'+'Plot Discovery params '+'.png')
    else:
        plt.show()

