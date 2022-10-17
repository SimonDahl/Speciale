

# v4 men kan tage flere inputs, den ser ikke ud til at lære forskellige løsning, 
# har leget en del med z dim, er i tvivl om phy res fungere helt som den skal 

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
import argparse
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from scipy.integrate import odeint

n_data = 10
bs = 1
time_limit = 10
n_col = 200
m = 1
k = 1
n_neurons = 50
lr = 0.001
lam = 1.5
lamda_q = 0.5
drop = 0.0
z_dim = 50
x_dim = n_col
y_dim = x_dim 
criterion = nn.BCELoss() 
criterion_mse = nn.MSELoss()
n_epochs = 100
gen_epoch = 5
np.random.seed(1234)
#y_data = -k*np.cos()+k
x_data = np.linspace(0,time_limit,n_col)

def pend(x, t, m, k):
    x1,x2 = x
    dxdt = [x2, -m*x2 - k*np.sin(x1)]
    return dxdt


t = np.linspace(0, time_limit,n_col)
   
u_sol = np.zeros((n_data,n_col))

for i in range(n_data):
    #m = np.random.uniform(1,5)
    x0 = [np.random.uniform(1,3),0]
    sol = odeint(pend, x0, t, args=(m, k))
    u_sol[i,:] = sol[:,0]
print('Data generation complete')

x_data = x_data.reshape(n_col,1)
x_data = Variable(torch.from_numpy(x_data).float(), requires_grad=True).to(device)
y_data = Variable(torch.from_numpy(u_sol).float(), requires_grad=True).to(device)


#ymean, ystd = torch.mean(y_data), torch.std(y_data)
#y_data = (y_data - ymean) / ystd

#x_plot = x_data.cpu().detach().numpy()
#y_plot = y_data[0,:].cpu().detach().numpy()
#plt.plot(x_plot,y_plot)
#plt.show()

#train_set = torch.from_numpy(y_data)

#train_set = train_set.float()

train_loader = torch.utils.data.DataLoader(dataset=y_data, batch_size=bs, shuffle=True)


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
        
    def getODEParam(self):
        
        return (self.m,self.k)
        
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
        return ((self.fc5(d)))  # sigmoid for probaility 
        

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
G_optimizer = optim.Adam(G.parameters(), lr=lr)
D_optimizer = optim.Adam(D.parameters(), lr=lr)
Q_optimizer = optim.Adam(Q.parameters(), lr=lr)



# Physics-Informed residual on the collocation points         
def compute_residuals(x,z):
   # m,k = G.getODEParam()
    g_input = torch.concat((x,z))
    u = G(g_input[:,-1])
    u_t = torch.autograd.grad(u.sum(), x_data, create_graph=True)[0]
    u_tt = torch.autograd.grad(u_t.sum(), x_data, create_graph=True)[0]
    r_ode = m*u_tt+k*x
    return r_ode 


def n_phy_prob(x):
    noise = Variable(torch.randn(z_dim,1).to(device))
    g_input = torch.concat((x,noise))
    u = G(g_input[:,-1])
    residual = compute_residuals(x,noise)
    return residual,u,noise



def G_train(x,y_train):
    
        
    
    for g_epoch in range(gen_epoch):
        
        G.zero_grad()
        
        phy_loss1,_,_ = n_phy_prob(x)

        #z = Variable(torch.randn(z_dim,1).to(device))
        #y_GAN = Variable(torch.ones(1).to(device))

        res,y_pred,G_noise = n_phy_prob(x)


        fake_logits_u = D(torch.concat((x[:,-1],y_pred)))

        z_pred = Q(torch.concat((x[:,-1],y_pred)))

        mse_loss_z = criterion_mse(z_pred,G_noise[:,-1])

        mse_loss = criterion_mse(y_pred,y_train[:,-1])
        
        #log_q = torch.mean(torch.square(z_pred-z))
        #print(fake_logits_u.shape)
       
        
        #phy_loss = torch.mean(phy_loss1**2)

        G_loss = fake_logits_u + mse_loss + mse_loss_z


        G_loss.backward(retain_graph=True)
        G_optimizer.step()

        
        return G_loss.data.item()

def discriminator_loss(logits_real_u, logits_fake_u):
        loss =  - torch.mean(torch.log(1.0 - torch.sigmoid(logits_real_u) + 1e-8) + torch.log(torch.sigmoid(logits_fake_u) + 1e-8)) 
        return loss

    



def Q_train(x):
    Q_optimizer.zero_grad()
    Q_noise = torch.randn(z_dim,1).to(device)
    g_input = torch.concat((x,Q_noise))
    y_pred = G(g_input[:,-1])
    z_pred = Q(torch.concat((x[:,-1],y_pred)))
    Q_loss = criterion_mse(z_pred,Q_noise[:,-1])
    Q_loss.backward()
    Q_optimizer.step()
    
    return Q_loss.data.item()

def D_train(x,y_train):
    
    
    D_optimizer.zero_grad()

    
    d_input = torch.concat((x,y_train))
    
    real_logits = D(d_input[:,-1])
    
    _,u,_ = n_phy_prob(x)
    fake_logits_u = D(torch.concat((x[:,-1],u)))

    D_loss = discriminator_loss(real_logits,fake_logits_u)
    D_loss.backward(retain_graph=True)
    D_optimizer.step()
    return D_loss.data.item()


#%% 
for epoch in range(1, n_epochs+1):           
    D_losses, G_losses,Q_losses = [], [],[]
    
    for batch_idx,y_train in enumerate(train_loader):
        y_train = y_train.T
        #x_plot = x_data.cpu().detach().numpy()
        #y_plot = y_train[:,-1].cpu().detach().numpy()
        #plt.plot(x_plot,y_plot)
        #plt.title('batch_idx'+str(batch_idx)+'epoch'+str(epoch))
        #plt.show()
        D_losses.append(D_train(x_data,y_train))
        G_losses.append(G_train(x_data,y_train))
        Q_losses.append(Q_train(x_data))

    print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
            (epoch), n_epochs, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))
                                                                                                                    
                                        

#%%
""" 
x_plot = x_data.cpu().detach().numpy()
y_plot = y_train[:,-1].cpu().detach().numpy()
plt.plot(x_plot,y_plot)
      #plt.title('batch_idx'+str(batch_idx)+'epoch'+str(epoch))
plt.show()
 """

with torch.no_grad():
    
    
    for i in range(5):
        
        z = Variable(torch.randn(z_dim).to(device))
        g_gen = torch.concat((x_data[:,-1],z))
        generated = G(g_gen)
        y = generated.cpu().detach().numpy()
        plt.plot(x_data[:,-1],y)
    plt.show()



""""
 with torch.no_grad():
    
    fig, ax = plt.subplots(2,4)
    fig.set_figheight(10)
    fig.set_figwidth(20)
    #fig.tight_layout()
    
    
    c = 1
    for i in range(0,2):
        for j in range(0,4):
            z = Variable(torch.randn(z_dim).to(device))
            g_gen = torch.concat((x_data[:,-1],z))
            generated = G(g_gen)
            y = generated.cpu().detach().numpy()
            ax[i,j].plot(x_data[:,-1],y)
            ax[i,j].set_title('Sample ' + str(c))
            c+= 1
    
    
    #fig.suptitle('n_epochs ' +str(n_epochs)+' z_dim_size '+str(z_dim)+' lr '+str(lr),fontsize="x-large")
    plt.show()
    
    #plt.savefig('./output/GAN/Pendulum/'+'n_epochs ' +str(n_epochs)+' z_dim_size '+str(z_dim)+' lr '+str(lr)+'.png')     
"""                                                                                                                        
# %%
