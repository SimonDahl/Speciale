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

parser = argparse.ArgumentParser()

parser.add_argument('--n_epochs',help='Number of epochs',type=int)
parser.add_argument('--z_dim_size',help='Number of z dims',type=int)
parser.add_argument('--lr',help='Learning rate',type=float)


args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#%%
# batch size 
bs = 100
n_epochs = args.n_epochs
#n_epochs = 100
# latent space size 
z_dim_size = args.z_dim_size
#z_dim_size = 3
#np.random.seed(2022)
lr = args.lr   #3e-4
#lr = 3e-4

n_data = 1500
t_gen = 1000
timesteps = t_gen*2

#%% Data generation 

def lv(X, t, alpha, beta, delta, gamma):
    x, y = X
    dotx = x * (alpha - beta * y)
    doty = y * (-delta + gamma * x)
    return np.array([dotx, doty])

t = np.linspace(0.,10,t_gen)
x = np.zeros((n_data,timesteps))

alpha =1.5 
beta = 1
delta = 3
gamma = 1

for i in range(n_data):
    #alpha = np.random.uniform(1,5) 
    #beta = np.random.uniform(1,5)
    #delta = np.random.uniform(1,5)
    #gamma = np.random.uniform(1,5)
    x0 =  np.random.uniform(1,5)
    y0 = np.random.uniform(1,5)
    X0 = [x0, y0]
    res = odeint(lv, X0, t, args = (alpha, beta, delta, gamma))
    x[i,:] = res.T.flatten()



# split into test, validation, and training sets
x_temp, x_test, _, _ = train_test_split(x, x, test_size=0.05)
x_train, x_valid, _, _ = train_test_split(x_temp,
                                          x_temp,
                                          test_size=0.1)
n_train = len(x_train)
n_valid = len(x_valid)
n_test = len(x_test)

#print(n_train)


train_set = torch.from_numpy(x_train)
test_set = torch.from_numpy(x_test)

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


#%%

optimizer = optim.Adam(vae.parameters(),lr=lr)
def loss_function(recon_x, x, mu, log_var):
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
        loss = loss_function(recon_batch, data, mu, log_var)
        
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
            test_loss += loss_function(recon, data, mu, log_var).item()
        
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


plt.plot(t,yp1[0:t_gen],label='Original Prey')
plt.plot(t,yp1[t_gen:],label='Original Predator')

encoded = vae.encoder(dp)
p = torch.stack(encoded,dim=0)

sample = vae.sampling(p[0],p[1])


decoded = vae.decoder(sample).cpu()

y = decoded.detach().numpy()


plt.plot(t,y[0:t_gen].flatten(),label='Decoded Prey')
plt.plot(t,y[t_gen:].flatten(),label='Decoded Predator')
plt.legend(loc='upper right')
plt.savefig('./output/VAE/LV/'+'Encode_Decode n_epochs ' +str(n_epochs)+' z_dim_size '+str(z_dim_size)+' lr '+str(lr)+'.png')
#plt.show()


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
            ax[i,j].plot(t,y[0:t_gen].flatten())
            ax[i,j].plot(t,y[t_gen:].flatten())
            ax[i,j].set_title('Sample ' + str(c))
            c+= 1


    fig.suptitle('n_epochs ' +str(n_epochs)+' z_dim_size '+str(z_dim_size)+' lr '+str(lr),fontsize="x-large")
    plt.savefig('./output/VAE/LV/'+'n_epochs ' +str(n_epochs)+' z_dim_size '+str(z_dim_size)+' lr '+str(lr)+'.png')
    #plt.show()

