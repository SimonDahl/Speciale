# Den her virker til en løsning 

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
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from scipy.integrate import odeint
parser = argparse.ArgumentParser()

parser.add_argument('--n_epochs',help='Number of epochs',type=int)
parser.add_argument('--weigth_clipping',help='Weigth Clipping 1 = on 0 = off',type=int)
parser.add_argument('--GP_on',help='GP on or off',type=int)
parser.add_argument('--n_sols',help='amount of solutions',type=int)
parser.add_argument('--optimizer_choice',help='0 is adam 1 is rmsprop',type=int)

args = parser.parse_args()



n_sols = 200
bs = 10
n_col = 100
n_neurons = 50
lr = 0.001
drop = 0.0
n_epochs = 200
gen_epoch = 1
d_epochs = 1
clip_value = 1
lambda_GP = 1 
optimizer_choice = 1 # RMSE prog 
weight_clipping = 1 
GP_on = 0

HPC = False 

if HPC == True:
    n_epochs = args.n_epochs
    weight_clipping = args.weigth_clipping
    GP_on = args.GP_on
    n_sols = args.n_sols
    optimizer_choice = args.optimizer_choice 
    
    

z_dim = 1
x_dim = 1
y_dim = 1 

criterion = nn.BCELoss() 
criterion_mse = nn.MSELoss()



lambda_phy = 0.08
lambda_q = 0.4
lambda_val = 0.05

time_limit = 6
timesteps = 100
n_data = timesteps
t = np.linspace(0,time_limit,timesteps)

#%% Genarate data


m = 2
k = 5
c = 1

def pend(x, t, m, k,c):
    x1,x2 = x
    dxdt = [x2, (-(c/m)*x2-(k/m)*x1)]
    return dxdt



idx = list(range(0,timesteps))
y_train = y_train = np.zeros((n_sols,n_data))
x_train = np.zeros((n_sols,n_data))

x_b = list(t[i] for i in idx)
x_b = np.array(x_b)

for i in range(n_sols):
    x0 = [np.random.uniform(1,10),np.random.uniform(-2,2)]
    solution = odeint(pend, x0, t, args=(m, k,c))
    sol_data = solution[:,0]
    sol_data = list(sol_data[i] for i in idx)
    sol_data = np.array([sol_data])
    y_train[i,:] = sol_data
    x_train[i,:] = x_b



""" for i in range(n_sols):
    plt.plot(t,y_train[i,:])
plt.title('Range of solutions')
plt.xlabel('t [s]')
plt.show()  """  

 
x_col = np.linspace(0, time_limit, n_col)
t_sample = t.reshape(timesteps,1)


x_b = Variable(torch.from_numpy(x_train).float(), requires_grad=True).to(device)
y_b = Variable(torch.from_numpy(y_train).float(), requires_grad=True).to(device)
y_b = y_b.reshape((n_sols,n_data))
x_b = x_b.reshape((n_sols,n_data))


X_star_norm = Variable(torch.from_numpy(t_sample).float(), requires_grad=True).to(device)
x_col = x_col.reshape(n_col,1)
x_col = Variable(torch.from_numpy(x_col).float(), requires_grad=True).to(device)

start = time.time() 


#%% Define network 

class Generator(nn.Module):
    def __init__(self, g_input_dim, g_output_dim):
        super(Generator, self).__init__()       
        self.fc1 = nn.Linear(g_input_dim, n_neurons)
        self.fc2 = nn.Linear(n_neurons, n_neurons)
        self.fc3 = nn.Linear(n_neurons, n_neurons)
        self.fc4 = nn.Linear(n_neurons, g_output_dim)
        
                
        
    # forward method
    def forward(self,y):
        y = torch.tanh(self.fc1(y)) #
        y = torch.tanh(self.fc2(y)) 
        y = torch.tanh(self.fc3(y)) 
        return self.fc4(y) 
    
  
    
class Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, n_neurons)
        self.fc2 = nn.Linear(n_neurons,n_neurons)
        self.fc3 = nn.Linear(n_neurons, 1)  # output dim = 1 for binary classification 
    
    # forward method
    def forward(self, d):
        d = torch.tanh(self.fc1(d))
        d = F.dropout(d, drop)
        d = torch.tanh(self.fc2(d))
        d = F.dropout(d, drop)
              
        return ((self.fc3(d))) 
        

#%% 

# build network
G = Generator(g_input_dim = z_dim+x_dim, g_output_dim = 1).to(device)
D = Discriminator(x_dim+y_dim+1).to(device)



# set optimizer 

if optimizer_choice == 0: 
    G_optimizer = optim.RMSprop(G.parameters(), lr=lr)
    D_optimizer = optim.RMSprop(D.parameters(), lr=lr)
    
if optimizer_choice == 1:
    G_optimizer = optim.Adam(G.parameters(), lr=lr)
    D_optimizer = optim.Adam(D.parameters(), lr=lr)
   

#%%


# Physics-Informed residual on the collocation points         
def compute_residuals(x,u):           
   
    u_t  = torch.autograd.grad(u, x, torch.ones_like(u), retain_graph=True,create_graph=True)[0]# computes dy/dx
    u_tt = torch.autograd.grad(u_t,  x, torch.ones_like(u_t),retain_graph=True ,create_graph=True)[0]# computes d^2y/dx^2
      
    r_ode = m*u_tt+c*u_t + k*u
          
    return r_ode



def n_phy_prob(x):
    noise = Variable(torch.randn(x.shape).to(device))
    g_input = torch.cat((x,noise),dim=1)
    u = G(g_input)
    res = compute_residuals(x,u)
    n_phy = torch.exp(-lambda_val * (res**2))
    return u,noise,res,n_phy



def discriminator_loss(logits_real_u, logits_fake_u, logits_fake_f, logits_real_f):
        loss =   torch.mean(logits_real_u) - torch.mean(logits_fake_u) \
                + torch.mean(logits_real_f) - torch.mean(logits_fake_f) 
        return loss


def generator_loss(logits_fake_u, logits_fake_f):
    gen_loss = torch.mean(logits_fake_u) + torch.mean(logits_fake_f)
    return gen_loss
 
 
 
def compute_gradient_penalty(y_real, y_pred,x):
    alpha = Variable(torch.rand(1).to(device))
    interpolates = (alpha * y_real + ((1 - alpha) * y_pred)).requires_grad_(True)
    
    d_input = torch.cat((x,interpolates),dim=1)
    d_interpolates = D(d_input)

    fake = Variable(torch.Tensor(y_real.shape[0], 1).fill_(1.0), requires_grad=False).to(device)

    gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates, grad_outputs=fake, create_graph=True, retain_graph=True, only_inputs=True,)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


#%% Discriminator training loop  
    
def D_train(x,y_train):
    
    for d_e in range(d_epochs):
    
        D_optimizer.zero_grad()
        D_loss = torch.zeros(bs)
        
        for i in range(bs):
                
                
            x_i = x[i,:]
            y_i = y_train[i,:]
            
            #plt.plot(x_i.cpu().detach().numpy(),y_i.cpu().detach().numpy())
            
            
        
            x_i = x_i.reshape((n_data,1))
            y_i = y_i.reshape((n_data,1))
            
                
            
            # real y value for Discriminator  
            real_prob = torch.ones_like(x_i)
             
            d_input = torch.cat((x_i,y_i,real_prob),dim=1)
            real_logits = D(d_input)
            
            
              # physics loss for boundary point 
            u,_,n_phy,_ = n_phy_prob(x_i)
   
            fake_logits_u = D(torch.cat((x_i,u,n_phy),dim=1))
            
                # physics loss for collocation points 
    
            u_col,_,n_phy_col,_ = n_phy_prob(x_col)
            fake_logits_col = D(torch.cat((x_col,u_col,n_phy_col),dim=1))
            
            
             # computing synthetic real logits on collocation points for discriminator loss

            real_prob_col = torch.ones_like(x_col)
            real_logits_col = D(torch.cat((x_col,u_col,real_prob_col),dim=1))
    
            D_loss[i] = discriminator_loss(real_logits,fake_logits_u,fake_logits_col,real_logits_col)
     
     
           
            if GP_on == 1:
                GP = compute_gradient_penalty(real_logits,fake_logits_u,x_i)
                D_loss[i] = D_loss[i] - lambda_GP * GP
        
        D_loss = torch.mean(D_loss)    
        D_loss.backward(retain_graph=True)
        D_optimizer.step()
        
        if weight_clipping ==1:
            for p in D.parameters():
                p.data.clamp_(-clip_value,clip_value)
    return D_loss.data.item()

#%% Generator training loop 

def G_train(x,y_train):

    for g_epoch in range(gen_epoch):
        
        G_loss = torch.zeros(bs)
        
        for i in range(bs):
        
            G.zero_grad()

            x_i = x[i,:]
            x_i = x_i.reshape((n_data,1))
            y_i = y_train[i,:]
            y_i = y_i.reshape((n_data,1))
        
            #physics loss for collocation points
            
            # u,noise,n_phy,res
            
            u_col,_,n_phy,phyloss_1  = n_phy_prob(x_col)
            fake_logits_col = D(torch.cat((x_col,u_col,n_phy),dim=1))


            # physics loss for boundary points 
       
            y_pred,G_noise,n_phy,phyloss2 = n_phy_prob(x_i)
            fake_logits_u = D(torch.cat((x_i,y_pred,n_phy),dim=1))

            
            mse_loss = criterion_mse(y_pred,y_i)
            
            adv_loss = generator_loss(fake_logits_u,fake_logits_col)
            
            phy_loss = (phyloss_1**2).mean() 
            
            G_loss[i] = adv_loss + lambda_phy * phy_loss + mse_loss
        
        G_loss = torch.mean(G_loss)
        G_loss.backward(retain_graph=True)
        G_optimizer.step()

    
    return G_loss




#%% 
for epoch in range(1, n_epochs+1):
    D_losses, G_losses,Q_losses = [], [],[]
    batch = 0
    for i in range((n_sols//bs)):
                          
       
        x_batch = x_b[batch:(batch+bs),:]
        y_batch = y_b[batch:(batch+bs),:]
        
        D_losses.append(D_train(x_batch,y_batch))
        G_losses.append(G_train(x_batch,y_batch))
        
        batch += bs

    print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
            (epoch), n_epochs, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))
                                                                                                                    

end = time.time() 
#print("Time elapsed during the calculation:", end - start) 


#%% Show generated sample 
with torch.no_grad():
    
    
    for i in range(3):
        z = Variable(torch.rand(X_star_norm.shape).to(device))
        generated = G(torch.cat((X_star_norm,z),dim=1))
        y = generated.cpu().detach().numpy()
        plt.plot(t,y)
        
    
    #plt.plot(t,y_real)
    
    if HPC == True:
        plt.savefig('./output/GAN/Pendulum/'+'n_epochs ' +str(n_epochs)+' Weight_clipping '+str(weight_clipping)+' GP_on '+str(GP_on)+' Optimizer '+str(optimizer_choice)+' n_sols '+str(n_sols)+'.png') 
    else: 
        plt.show()

    

