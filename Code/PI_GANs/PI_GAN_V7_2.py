# er lavet helt som i artiklen, har både boundary points og tager kun et punkt af gangen som input. ser ikke ud til at lære noget som helst 


# hvis compute res er test equaion ligner det rent faktisk den, på trods af data er pendul 

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
time_limit = 5
n_col = 200
m = 1
k = 1


#y_data = np.cos(x_data*np.sqrt(k)) # Exact solution for (0,1) boundary condition
n_neurons = 50
lr = 0.001
drop = 0.0
z_dim = 1
x_dim = 1
y_dim = 1 
criterion = nn.BCELoss() 
criterion_mse = nn.MSELoss()
n_epochs = 300

gen_epoch = 5
lambda_phy = 2
lambda_q = 1
#y_data = -k*np.cos()+k
t = np.linspace(0, time_limit, n_col)

lam = 1


def pend(x, t, m, k):
    x1,x2 = x
    dxdt = [x2, -m*x2 - k*np.sin(x1)]
    return dxdt



sol_data = np.zeros((n_data,n_col))
y_b = np.zeros((n_data,1))

for i in range(n_data):
    #m = np.random.uniform(1,5)
    x0 = [np.random.uniform(1,3),0]
    sol = odeint(pend, x0, t, args=(m, k))
    sol_data[i,:] = sol[:,0]
    y_b[i] = x0[0]
print('Data generation complete')



x_col = np.linspace(0.1, time_limit, n_col)

x_b = np.zeros([1])
x_b = Variable(torch.from_numpy(x_b).float(), requires_grad=True).to(device)
y_b = Variable(torch.from_numpy(y_b).float(), requires_grad=True).to(device)

t_sample = t.reshape(n_col,1)
t_sample = Variable(torch.from_numpy(t_sample).float(), requires_grad=True).to(device)
x_col = x_col.reshape(n_col,1)
x_col = Variable(torch.from_numpy(x_col).float(), requires_grad=True).to(device)
#y_data = Variable(torch.from_numpy(y_data).float(), requires_grad=True).to(device)
# y_data = Variable(torch.from_numpy(sol_data).float(), requires_grad=True).to(device)
#y_data = y_data.reshape(n_col,n_data) 

sol_data = sol_data.T

#plt.plot(t,sol_data[:,-1])



class Generator(nn.Module):
    def __init__(self, g_input_dim, g_output_dim):
        super(Generator, self).__init__()       
        self.fc1 = nn.Linear(g_input_dim, n_neurons)
        self.fc2 = nn.Linear(n_neurons, n_neurons)
        self.fc3 = nn.Linear(n_neurons, n_neurons)
        self.fc4 = nn.Linear(n_neurons, g_output_dim)
    
        
        
    # forward method
    def forward(self,y):
        y = torch.tanh(self.fc1(y)) # leaky relu, with slope angle 
        y = torch.tanh(self.fc2(y)) 
        y = torch.tanh(self.fc3(y))
         #return torch.tanh(self.fc4(x))
        return self.fc4(y) 

    
class Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, n_neurons)
        self.fc2 = nn.Linear(n_neurons,n_neurons)
        self.fc3 = nn.Linear(n_neurons,n_neurons)
        self.fc4 = nn.Linear(n_neurons, 1)  # output dim = 1 for binary classification 
    
    # forward method
    def forward(self, d):
        d = torch.tanh(self.fc1(d))
        d = F.dropout(d, drop)
        d = torch.tanh(self.fc2(d))
        d = F.dropout(d, drop)
        d = torch.tanh(self.fc3(d))
        d = F.dropout(d, drop)
        
        return ((self.fc4(d)))  # sigmoid for probaility 
        

class Q_net(nn.Module):
    def __init__(self, Q_input_dim,Q_output_dim):
        super(Q_net, self).__init__()
        self.fc1 = nn.Linear(Q_input_dim, n_neurons)
        self.fc2 = nn.Linear(n_neurons,n_neurons)
        self.fc3 = nn.Linear(n_neurons,Q_output_dim)
    
    def forward(self,q):
     
        q = torch.tanh(self.fc1(q)) # leaky relu, with slope angle 
        q = torch.tanh(self.fc2(q))
        
        return (self.fc3(q))

# build network
G = Generator(g_input_dim = z_dim+x_dim, g_output_dim = y_dim).to(device)
D = Discriminator(x_dim+y_dim).to(device)
Q = Q_net(x_dim+y_dim,z_dim)


# set optimizer 
G_optimizer = optim.Adam(G.parameters(), lr=lr)
D_optimizer = optim.Adam(D.parameters(), lr=lr)
Q_optimizer = optim.Adam(Q.parameters(), lr=lr)



# Physics-Informed residual on the collocation points         
def compute_residuals(x_collocation):
    z = Variable(torch.randn(z_dim).to(device))
    r_ode = np.zeros([n_col])
    for i in range(n_col):
        x = x_collocation[i]
        u = G(torch.concat((x,z)))
        u_t = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_tt = torch.autograd.grad(u_t.sum(), x, create_graph=True)[0]
        r_ode[i]= m*u_tt+k*x
            
    res = np.mean(r_ode**2)
    return res,r_ode
def n_phy_prob(x):
    noise = Variable(torch.randn(z_dim).to(device))
    g_input = torch.concat((x,noise))
    u = G(g_input)
    return u,noise

def discriminator_loss(logits_real_u, logits_fake_u):
        loss =  - torch.mean(torch.log(1.0 - torch.sigmoid(logits_real_u) + 1e-8) + torch.log(torch.sigmoid(logits_fake_u) + 1e-8)) 
        return loss


def D_train(x,y_train):
    
    D_optimizer.zero_grad()
     
     
    # real y value for Discriminator  
    d_input = torch.concat((x,y_train))
    real_logits = D(d_input)
  
    # physics loss for boundary point 
    u,_ = n_phy_prob(x)
    fake_logits_u = D(torch.concat((x,u)))


    D_loss = discriminator_loss(real_logits,fake_logits_u)
    D_loss.backward(retain_graph=True)
    D_optimizer.step()
    return D_loss.data.item()


def G_train(x,y_train):

    for g_epoch in range(gen_epoch):
        
        G.zero_grad()

        #physics loss for collocation points
        
        phy_loss,_  = compute_residuals(x_col)

        # physics loss for boundary points 
        
        y_pred,G_noise = n_phy_prob(x)
        fake_logits_u = D(torch.concat((x,y_pred)))

        z_pred = Q(torch.concat((x,y_pred)))
        mse_loss_z = criterion_mse(z_pred,G_noise)

        mse_loss = criterion_mse(y_pred,y_train)
        adv_loss = fake_logits_u
        
        
        """ print(phy_loss)
        p_loss = criterion_mse(phy_loss,np.zeros_like(phy_loss)) """

        G_loss = adv_loss + lambda_phy* phy_loss + lambda_q * mse_loss_z


        G_loss.backward(retain_graph=True)
        G_optimizer.step()

    return G_loss, phy_loss


def Q_train(x):
    
    
    Q_optimizer.zero_grad()
    Q_noise = torch.randn(z_dim).to(device)
    g_input = torch.concat((x,Q_noise))
    y_pred = G(g_input)
    z_pred = Q(torch.concat((x,y_pred)))
    Q_loss = criterion_mse(z_pred,Q_noise)
    Q_loss.backward()
    Q_optimizer.step()
    
    return Q_loss.data.item()


#%% 
for epoch in range(1, n_epochs+1):
    D_losses, G_losses,Q_losses = [], [],[]

    for batch in range(bs):
        if n_data > 1:
            idx = np.random.randint(0,n_data)
            y_train = y_b[idx]       
            y_train = y_train[0]
        else: 
            y_train = y_b[0]
        
        D_losses.append(D_train(x_b,y_train))
        G_losses.append(G_train(x_b,y_train))
        Q_losses.append(Q_train(x_b))
        
        
    print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
            (epoch), n_epochs, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))
                                                                                                                    

# generata sample
""" 
u_plot = np.zeros(n_col)
z = Variable(torch.randn(z_dim).to(device))
for i in range(n_col):
    
    u_plot[i] = G(torch.concat((t_sample[i],z)))
    

plt.plot(t,u_plot)
plt.show() """

t_test = np.linspace(0, time_limit, n_col)
t_test = t_test.reshape(n_col,1)
t_test = Variable(torch.from_numpy(t_test).float(), requires_grad=True).to(device)


res,res_plot = compute_residuals(t_test)
t_plot = t_test.cpu().detach().numpy()
plt.plot(t_plot,res_plot)
plt.show()

""" 
u_plot = np.zeros(n_col)

for i in range(5):
    z = Variable(torch.randn(z_dim).to(device))
    for i in range(n_col):
        u_plot[i] = G(torch.concat((t_sample[i],z)))
    plt.plot(t,u_plot)
plt.show() 
 """
