from sqlite3 import Timestamp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.integrate import odeint
import argparse


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#%%
# batch size 
bs = 500
n_epochs = 3000
z_dim_size = 3

np.random.seed(1234)
lr = 0.001
timesteps = 500
n_data = 1000
time_limit = 1
n_col = timesteps


#%% Generate data 

t = np.linspace(0, time_limit, n_col)

k = 1
def test_equation(x, t,k):
    dxdt = k*x
    return dxdt

sol_data = np.zeros((n_data,n_col))



for i in range(n_data):
    x0 = np.random.uniform(1,3)
    sol = odeint(test_equation, x0, t, args=(k,))
    sol_data[i,:] = sol[:,-1]
   
x_col =  np.linspace(0, time_limit, n_col)
x_col = Variable(torch.from_numpy(x_col).float(), requires_grad=True).to(device)



y = sol_data 
print(np.isnan(y).any())




""" for i in range(10):
    
    plt.plot(t,y[i,:])
plt.show()
 """

#%%

# split into test, validation, and training sets
y_temp, y_test, _, _ = train_test_split(y, y, test_size=0.05)
y_train, y_valid, _, _ = train_test_split(y_temp,
                                          y_temp,
                                          test_size=0.1)
n_train = len(y_train)
n_valid = len(y_valid)
n_test = len(y_test)

#print(n_train)


train_set = torch.from_numpy(y_train)
test_set = torch.from_numpy(y_test)

train_set = train_set.float()
test_set = test_set.float()

train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=bs, shuffle=False)



#print(test_set.shape)
#%%

class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2,h_dim3,h_dim4,z_dim):
        super(VAE, self).__init__()
        
       
        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc3 = nn.Linear(h_dim2, h_dim3)
        self.fc4 = nn.Linear(h_dim3, h_dim4)
        self.fc4_sigma = nn.Linear(h_dim4, z_dim)
        self.fc4_mu = nn.Linear(h_dim4, z_dim)
        # decoder part
        self.fc_z = nn.Linear(z_dim, h_dim4)
        self.fc5 = nn.Linear(h_dim4, h_dim3)
        self.fc6 = nn.Linear(h_dim3, h_dim2)
        self.fc7 = nn.Linear(h_dim2, h_dim1)
        self.fc8 = nn.Linear(h_dim1, x_dim)
        
    def encoder(self, x):
        h = F.leaky_relu(self.fc1(x))
        h = F.leaky_relu(self.fc2(h))
        h = F.leaky_relu(self.fc3(h))
        h = F.leaky_relu(self.fc4(h))
        return self.fc4_sigma(h), self.fc4_mu(h) # mu, log_var
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def decoder(self, z):
        h = F.leaky_relu(self.fc_z(z))
        h = F.leaky_relu(self.fc5(h))
        h = F.leaky_relu(self.fc6(h))
        h = F.leaky_relu(self.fc7(h))
        return (self.fc8(h)) 
    
    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, timesteps))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var

h1= timesteps//2
h2 = h1//2
h3 = h2//2
h4 = h3//2

vae = VAE(x_dim=timesteps, h_dim1=h1, h_dim2=h2,h_dim3=h3,h_dim4=h4, z_dim=z_dim_size)


if torch.cuda.is_available():
    vae.cuda()


# Physics-Informed residual on the collocation points         
def compute_residuals(x_col):
   # m,k = G.getODEParam()
    
    z = vae.encoder(x_col)
    z = torch.stack(z,dim=0)

    sample = vae.sampling(z[0],z[1])
    u = vae.decoder(sample)
    u_t = torch.autograd.grad(u.sum(), x_col, create_graph=True)[0]
    res = k*x_col - u_t
    r_ode = torch.sum(res**2)
    return r_ode

#%%



optimizer = optim.Adam(vae.parameters(),lr=lr)
def loss_function(recon_x, x, mu, log_var,x_col):
    MSE = F.mse_loss(recon_x, x.view(-1, timesteps), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    r_ode = compute_residuals(x_col)
    return MSE + KLD + r_ode 
  
def test_loss_function(recon_x, x, mu, log_var):
    MSE = F.mse_loss(recon_x, x.view(-1, timesteps), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return MSE + KLD 
  
#%%

def train(epoch):
    vae.train()
    train_loss = 0
    for batch_idx, (data) in enumerate(train_loader):
        if torch.cuda.is_available():
            data = data.cuda()
        optimizer.zero_grad()
        
        recon_batch, mu, log_var = vae(data)
        loss = loss_function(recon_batch, data, mu, log_var,x_col)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        
    if epoch % 20 == 0:
      print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))
      
def test():
    vae.eval()
    test_loss= 0
    with torch.no_grad():
        for data in test_loader:
            if torch.cuda.is_available():
                data = data.cuda()
            recon, mu, log_var = vae(data)
            
            # sum up batch loss
            test_loss += test_loss_function(recon, data, mu, log_var).item()
        
    test_loss /= len(test_loader.dataset)
    if epoch % 20 == 0:
      print('====> Test set loss: {:.4f}'.format(test_loss))
      

for epoch in range(1, n_epochs):
    train(epoch)
    test()




#%%


data_point = []

for data in test_loader:
    if torch.cuda.is_available():        
        data = data.cuda()
    data_point.append(data)


dp = data_point[0][0,:]

yp1 = dp.cpu().detach().numpy()



plt.plot(t,yp1,label='Original')

encoded = vae.encoder(dp)
p = torch.stack(encoded,dim=0)

sample = vae.sampling(p[0],p[1])


decoded = vae.decoder(sample).cpu()

y = decoded.detach().numpy()


plt.plot(t,y.flatten(),label='Decoded')
plt.legend(loc='upper right')
#plt.savefig('./output/VAE/Test_Equation/'+'Encode_Decode n_epochs ' +str(n_epochs)+' z_dim_size '+str(z_dim_size)+' lr '+str(lr)+'.png')
plt.show()











with torch.no_grad():



    fig, ax = plt.subplots(2,4)
    fig.set_figheight(10)
    fig.set_figwidth(20)
    #fig.tight_layout()
    
    
    c = 1
    for i in range(0,2):
        for j in range(0,4):
            z = Variable(torch.randn(z_dim_size).to(device))
            if torch.cuda.is_available():
                sample = vae.decoder(z).cuda()
            else:
                sample = vae.decoder(z)
            y = sample.cpu().detach().numpy()
            ax[i,j].plot(t,y.flatten())
            ax[i,j].set_title('Sample ' + str(c))
            c+= 1


    fig.suptitle('n_epochs ' +str(n_epochs)+' z_dim_size '+str(z_dim_size)+' lr '+str(lr),fontsize="x-large")
    #plt.savefig('./output/VAE/Test_Equation/'+'n_epochs ' +str(n_epochs)+' z_dim_size '+str(z_dim_size)+' lr '+str(lr)+'.png')
     
    plt.show()
