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
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from plotting import newfig, savefig
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import argparse
import scipy
from itertools import product, combinations
from scipy.interpolate import griddata
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.nn.utils import weight_norm as wn
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

#%% Hyperparameters 

np.random.seed(1234)
n_neurons = 75
n_neurons_d = 128
lr = 0.001
criterion_MSE = nn.MSELoss() # loss function
criterion_BCE = nn.BCELoss() 
lambda_phy = 0.5
D_epochs = 1



#%% HPC and data load 

HPC = True

if HPC == True:
    print('Started code')
    n_epochs = 45000
    switch = 30000
    lr2 = 0.0005
    N_train = 5000
    data = scipy.io.loadmat('cylinder_nektar_wake.mat')
if HPC == False: 
    n_epochs = 20
    switch = 10000
    lr2 = 0.0001
    N_train = 50
    
    data = scipy.io.loadmat(r"C:\Users\Simon\OneDrive - Danmarks Tekniske Universitet\Speciale\Speciale\Code\NSGAN\cylinder_nektar_wake.mat")

bs = N_train//10 
lambda_1 = 1
lambda_2 = 0.01


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

    return p,u,v

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
  
    f_u = u_t + lambda_1*(u*u_x + v*u_y) + p_x - lambda_2*(u_xx + u_yy) 
    f_v = v_t + lambda_1*(u*v_x + v*v_y) + p_y - lambda_2*(v_xx + v_yy)
    
    return u, v, p, f_u, f_v


def rearange(x,y,t,p,u,v):
    img = torch.zeros(N_train,1,3,3)
    for i in range(N_train):
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
    p_fake,u_fake,v_fake = predict(x,y,t)
    
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
        
        p_fake,u_fake,v_fake = predict(x,y,t)
        d_input = rearange(x,y,t,p_fake,u_fake,v_fake)    
        fake_prob = D(d_input)
        

        d_input = rearange(x,y,t,p,u,v)
        real_prob = D(d_input)
        
        D_loss = torch.mean(-torch.log(real_prob+1e-8)+torch.log(1-fake_prob+1e-8))
                
        D_loss.backward()
        D_optimizer.step()
    
    return D_loss

#%%& Training loop 

for epoch in range(1, n_epochs+1):
    D_losses, G_losses, MSE_losses = [], [], []

  #  if epoch == switch:
   #     G_optimizer = optim.Adam(G.parameters(), lr=lr2)
    #    D_optimizer = optim.Adam(D.parameters(), lr=lr2)


    #for batch_idx in range(N_train//bs):
     #   idx = np.random.choice(N_train, bs, replace=False)
      #  x_batch = x_train[idx,:]
       # y_batch = x_train[idx,:]
        #t_batch = x_train[idx,:]
      #  u_batch = x_train[idx,:]
      #  v_batch = x_train[idx,:]
      #  p_batch = x_train[idx,:]
    G_loss,MSE_loss =  G_train(x_train,y_train,t_train,p_train,u_train,v_train)
    G_losses.append(G_loss)
    MSE_losses.append(MSE_loss)
    D_losses.append(D_train(x_train,y_train,t_train,p_train,u_train,v_train))

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
X_star = Variable(torch.from_numpy(X_star).float(), requires_grad=True).to(device)


p_pred,u_pred, v_pred = predict(x_star, y_star, t_star)

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
        plt.title('Pressure error')

    if HPC == False: 
        plt.show()
    if HPC == True: 
        plt.savefig('./output/NSGAN/'+'Plot ' +str(index)+'.png')
         


def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/4
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

#%% Plot and print results 

with torch.no_grad():
    x_star = x_star.cpu().detach().numpy()
    y_star = y_star.cpu().detach().numpy()
    u_star = u_star.cpu().detach().numpy()
    u_pred = u_pred.cpu().detach().numpy()
    v_star = v_star.cpu().detach().numpy()
    v_pred = v_pred.cpu().detach().numpy()
    p_star = p_star.cpu().detach().numpy()
    p_pred = p_pred.cpu().detach().numpy()
    x_train = x_train.cpu().detach().numpy()
    t_train = t_train.cpu().detach().numpy()
    p_train = p_train.cpu().detach().numpy()
    y_train = y_train.cpu().detach().numpy()
    X_star = X_star.cpu().detach().numpy()
    t_star = t_star.cpu().detach().numpy()

    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    error_v = np.linalg.norm(v_star-v_pred,2)/np.linalg.norm(v_star,2)
    error_p = np.linalg.norm(p_star-p_pred,2)/np.linalg.norm(p_star,2)
        
    print('Error u: %e' % (error_u))    
    print('Error v: %e' % (error_v))    
    print('Error p: %e' % (error_p))    

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
    


######################################################################
    ############################# Plotting ###############################
    ######################################################################    
     # Load Data
    if HPC == False:
        data_vort = scipy.io.loadmat(r"C:\Users\Simon\OneDrive - Danmarks Tekniske Universitet\Speciale\Speciale\Code\NSGAN\/cylinder_nektar_t0_vorticity.mat")
    else: 
        data_vort = scipy.io.loadmat('cylinder_nektar_t0_vorticity.mat')       
    x_vort = data_vort['x'] 
    y_vort = data_vort['y'] 
    w_vort = data_vort['w'] 
    modes = (data_vort['modes']).item()
    nel = (data_vort['nel']).item()        
    
    xx_vort = np.reshape(x_vort, (modes+1,modes+1,nel), order = 'F')
    yy_vort = np.reshape(y_vort, (modes+1,modes+1,nel), order = 'F')
    ww_vort = np.reshape(w_vort, (modes+1,modes+1,nel), order = 'F')
    
    box_lb = np.array([1.0, -2.0])
    box_ub = np.array([8.0, 2.0])
    
    fig, ax = newfig(1.0, 1.2)
    ax.axis('off')
    
    ####### Row 0: Vorticity ##################    
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-2/4 + 0.12, left=0.0, right=1.0, wspace=0)
    ax = plt.subplot(gs0[:, :])
    
    for i in range(0, nel):
        h = ax.pcolormesh(xx_vort[:,:,i], yy_vort[:,:,i], ww_vort[:,:,i], cmap='seismic',shading='gouraud',  vmin=-3, vmax=3) 
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    
    ax.plot([box_lb[0],box_lb[0]],[box_lb[1],box_ub[1]],'k',linewidth = 1)
    ax.plot([box_ub[0],box_ub[0]],[box_lb[1],box_ub[1]],'k',linewidth = 1)
    ax.plot([box_lb[0],box_ub[0]],[box_lb[1],box_lb[1]],'k',linewidth = 1)
    ax.plot([box_lb[0],box_ub[0]],[box_ub[1],box_ub[1]],'k',linewidth = 1)
    
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('Vorticity', fontsize = 10)
    #plt.show()
    
    ####### Row 1: Training data ##################
    ########      u(t,x,y)     ###################        
    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(top=1-2/4, bottom=0.0, left=0.01, right=0.99, wspace=0)
    ax = plt.subplot(gs1[:, 0],  projection='3d')
    ax.axis('off')

    r1 = [x_star.min(), x_star.max()]
    r2 = [data['t'].min(), data['t'].max()]       
    r3 = [y_star.min(), y_star.max()]

    
    for s, e in combinations(np.array(list(product(r1,r2,r3))), 2):
        if np.sum(np.abs(s-e)) == r1[1]-r1[0] or np.sum(np.abs(s-e)) == r2[1]-r2[0] or np.sum(np.abs(s-e)) == r3[1]-r3[0]:
            ax.plot3D(*zip(s,e), color="k", linewidth = 0.5)   

    ax.scatter(x_train, t_train, y_train, s = 0.1)
    ax.contourf(X,UU_star,Y, zdir = 'y', offset = t_star.mean(), cmap='rainbow', alpha = 0.8)
              
    ax.text(x_star.mean(), data['t'].min() - 1, y_star.min() - 1, '$x$')
    ax.text(x_star.max()+1, data['t'].mean(), y_star.min() - 1, '$t$')
    ax.text(x_star.min()-1, data['t'].min() - 0.5, y_star.mean(), '$y$')
    ax.text(x_star.min()-3, data['t'].mean(), y_star.max() + 1, '$u(t,x,y)$')    
    ax.set_xlim3d(r1)
    ax.set_ylim3d(r2)
    ax.set_zlim3d(r3)
    axisEqual3D(ax)
    
    ########      v(t,x,y)     ###################        
    ax = plt.subplot(gs1[:, 1],  projection='3d')
    ax.axis('off')
    
    r1 = [x_star.min(), x_star.max()]
    r2 = [data['t'].min(), data['t'].max()]       
    r3 = [y_star.min(), y_star.max()]


    for s, e in combinations(np.array(list(product(r1,r2,r3))), 2):
        if np.sum(np.abs(s-e)) == r1[1]-r1[0] or np.sum(np.abs(s-e)) == r2[1]-r2[0] or np.sum(np.abs(s-e)) == r3[1]-r3[0]:
            ax.plot3D(*zip(s,e), color="k", linewidth = 0.5)   

    ax.scatter(x_train, t_train, y_train, s = 0.1)
    ax.contourf(X,VV_star,Y, zdir = 'y', offset = t_star.mean(), cmap='rainbow', alpha = 0.8)
              
    ax.text(x_star.mean(), data['t'].min() - 1, y_star.min() - 1, '$x$')
    ax.text(x_star.max()+1, data['t'].mean(), y_star.min() - 1, '$t$')
    ax.text(x_star.min()-1, data['t'].min() - 0.5, y_star.mean(), '$y$')
    ax.text(x_star.min()-3, data['t'].mean(), y_star.max() + 1, '$v(t,x,y)$')    
    ax.set_xlim3d(r1)
    ax.set_ylim3d(r2)
    ax.set_zlim3d(r3)
    axisEqual3D(ax)
    if HPC == True:
        plt.savefig('./output/NSGAN/'+'3D'+'.png')
    else:
        plt.show()
    # savefig('./figures/NavierStokes_data') 

    
    fig, ax = newfig(1.015, 0.8)
    ax.axis('off')
    
    ######## Row 2: Pressure #######################
    ########      Predicted p(t,x,y)     ########### 
    gs2 = gridspec.GridSpec(1, 2)
    gs2.update(top=1, bottom=1-1/2, left=0.1, right=0.9, wspace=0.5)
    ax = plt.subplot(gs2[:, 0])
    h = ax.imshow(PP_star, interpolation='nearest', cmap='rainbow', 
                  extent=[x_star.min(), x_star.max(), y_star.min(), y_star.max()], 
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_aspect('equal', 'box')
    ax.set_title('Predicted pressure', fontsize = 10)
    
    ########     Exact p(t,x,y)     ########### 
    ax = plt.subplot(gs2[:, 1])
    h = ax.imshow(P_exact, interpolation='nearest', cmap='rainbow', 
                  extent=[x_star.min(), x_star.max(), y_star.min(), y_star.max()], 
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_aspect('equal', 'box')
    ax.set_title('Exact pressure', fontsize = 10)
    if HPC == True:
        plt.savefig('./output/NSGAN/'+'Predict_vs_exact'+'.png')
    else:
        plt.show()
    
"""     ######## Row 3: Table #######################
    gs3 = gridspec.GridSpec(1, 2)
    gs3.update(top=1-1/2, bottom=0.0, left=0.0, right=1.0, wspace=0)
    ax = plt.subplot(gs3[:, :])
    ax.axis('off') 
   
     s = r'$\begin{tabular}{|c|c|}';
    s = s + r' \hline'
    s = s + r' Correct PDE & $\begin{array}{c}'
    s = s + r' u_t + (u u_x + v u_y) = -p_x + 0.01 (u_{xx} + u_{yy})\\'
    s = s + r' v_t + (u v_x + v v_y) = -p_y + 0.01 (v_{xx} + v_{yy})'
    s = s + r' \end{array}$ \\ '
    s = s + r' \hline'
    s = s + r' Identified PDE (clean data) & $\begin{array}{c}'
    s = s + r' u_t + %.3f (u u_x + v u_y) = -p_x + %.5f (u_{xx} + u_{yy})' % (lambda_1_value, lambda_2_value)
    s = s + r' \\'
    s = s + r' v_t + %.3f (u v_x + v v_y) = -p_y + %.5f (v_{xx} + v_{yy})' % (lambda_1_value, lambda_2_value)
    s = s + r' \end{array}$ \\ '
    s = s + r' \hline'
    s = s + r' Identified PDE (1\% noise) & $\begin{array}{c}'
    s = s + r' u_t + %.3f (u u_x + v u_y) = -p_x + %.5f (u_{xx} + u_{yy})' % (lambda_1_value_noisy, lambda_2_value_noisy)
    s = s + r' \\'
    s = s + r' v_t + %.3f (u v_x + v v_y) = -p_y + %.5f (v_{xx} + v_{yy})' % (lambda_1_value_noisy, lambda_2_value_noisy)
    s = s + r' \end{array}$ \\ '
    s = s + r' \hline'
    s = s + r' \end{tabular}$' """
 
   # ax.text(0.015,0.0,s)
    
    # savefig('./figures/NavierStokes_prediction') 