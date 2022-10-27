

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

n_data = 50
bs = 1
time_limit = 5
n_col = 200
m = 1
k = 1
n_neurons = 50
lr = 0.001
lam = 1.5
lamda_q = 0.5
drop = 0.0
z_dim = 100
x_dim = n_col
y_dim = x_dim 
criterion = nn.BCELoss() 
criterion_mse = nn.MSELoss()
n_epochs = 500
gen_epoch = 5

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
    x0 = [np.random.uniform(1,10),np.random.uniform(-2,2)]
    sol = odeint(pend, x0, t, args=(m, k))
    u_sol[i,:] = sol[:,0]
print('Data generation complete')

x_data = x_data.reshape(n_col,1)
x_data = Variable(torch.from_numpy(x_data).float(), requires_grad=True).to(device)
y_data = Variable(torch.from_numpy(u_sol).float(), requires_grad=True).to(device)


#ymean, ystd = torch.mean(y_data), torch.std(y_data)
#y_data = (y_data - ymean) / ystd

x_plot = x_data.cpu().detach().numpy()
""" 
for i in range(10):
    y_plot = y_data[i,:].cpu().detach().numpy()
    plt.plot(x_plot,y_plot)
plt.show()
 """
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
    
        
        
    def getODEParam(self):
        
        return (self.m,self.k)
        
    # forward method
    def forward(self,y):
        y = torch.relu(self.fc1(y)) # leaky relu, with slope angle 
        y = torch.relu(self.fc2(y)) 
        y = torch.relu(self.fc3(y))
        y = torch.relu(self.fc4(y))
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
        


# build network
G = Generator(g_input_dim = z_dim+x_dim, g_output_dim = y_dim).to(device)
D = Discriminator(x_dim+y_dim).to(device)


# set optimizer 
G_optimizer = optim.Adam(G.parameters(), lr=lr)
D_optimizer = optim.Adam(D.parameters(), lr=lr)




# Physics-Informed residual on the collocation points         
def compute_residuals(x,z):
   # m,k = G.getODEParam()
    
    g_input = torch.concat((x,z))
    
    u = G(g_input[:,-1])
    
    u_t = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    #u_tt = torch.autograd.grad(u_t.sum(), x, create_graph=True)[0]
    #r_ode = m*u_tt+k*x
    r_ode = u*k - u_t
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

        res,y_pred,G_noise = n_phy_prob(x)


        fake_logits_u = D(torch.concat((x[:,-1],y_pred)))

        
        

        mse_loss = criterion_mse(y_pred,y_train[:,-1])
        
        #log_q = torch.mean(torch.square(z_pred-z))
        #print(fake_logits_u.shape)
       
        
        phy_loss = torch.mean(phy_loss1**2)

        G_loss = fake_logits_u + mse_loss + phy_loss


        G_loss.backward(retain_graph=True)
        G_optimizer.step()

        
        return G_loss.data.item()

def discriminator_loss(logits_real_u, logits_fake_u):
        loss =  - torch.mean(torch.log(1.0 - torch.sigmoid(logits_real_u) + 1e-8) + torch.log(torch.sigmoid(logits_fake_u) + 1e-8)) 
        return loss

    


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
    D_losses, G_losses, = [], []
    
    for batch_idx,y_train in enumerate(train_loader):
        y_train = y_train.T
        #x_plot = x_data.cpu().detach().numpy()
        #y_plot = y_train[:,-1].cpu().detach().numpy()
        #plt.plot(x_plot,y_plot)
        #plt.title('batch_idx'+str(batch_idx)+'epoch'+str(epoch))
        #plt.show()
        D_losses.append(D_train(x_data,y_train))
        G_losses.append(G_train(x_data,y_train))
       
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


t_test = np.linspace(0, time_limit, n_col)
t_test = t_test.reshape(n_col,1)
t_test = Variable(torch.from_numpy(t_test).float(), requires_grad=True).to(device)

z_test = Variable(torch.randn(z_dim,1).to(device))
res = compute_residuals(t_test,z_test)
res_plot = res.cpu().detach().numpy()
t_plot = t_test.cpu().detach().numpy()
plt.plot(t_plot,res_plot)
plt.show()


with torch.no_grad():
    
    
    for i in range(5):
        
        z = Variable(torch.randn(z_dim).to(device))
        
        g_gen = torch.concat((x_data[:,-1],z))
        generated = G(g_gen)
        print(torch.mean(generated))
        y = generated.cpu().detach().numpy()
        plt.plot(x_data[:,-1],y)
    plt.show()

 
""" with torch.no_grad(): 
 
    z =  Variable(torch.randn(z_dim).to(device))
    z2 = torch.zeros_like(z)
    x_test = torch.zeros_like(x_data) 
    print(x_data,z)
    print(x_test,z2)
    g_gen = torch.concat((x_data[:,-1],z))
    g_gen1 = torch.concat((x_test[:,-1],z2))
    gen = G(g_gen)
    gen1 = G(g_gen1)
    y = gen.cpu().detach().numpy()
    y1 = gen1.cpu().detach().numpy()

    plt.plot(x_data[:,-1],y)
    plt.plot(x_data,y1)

    plt.show()
 """
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