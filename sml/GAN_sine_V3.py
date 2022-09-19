# prerequisites
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
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%% Hyperparamters 

# Batch size 
bs = 300
# Dimension of z - noise vector 
z_dim = 100

# loss 
criterion = nn.BCELoss() 
# learning rate
lr = 0.0002 
# number of epochs 
n_epoch = 1

#%% Generate sine wave data

def sin_func(x):
  return np.sin(x)

n = 30000              # number of waves
nt = 128*4              # time steps pr wave 
#f = 3.0                  # frequency in Hz



t = np.linspace(0,1,nt)  # time stamps in s
x = np.zeros((n,nt))
phase = np.random.uniform(-np.pi, np.pi, size=n)
for i in range(n):
    f = np.random.uniform(1,5) # frequency in Hz  
    A = np.random.uniform(1,5) # random amplitude
    x[i,:] = A*np.sin(2*np.pi*f*t + phase[i] )
   
   
#%% Ready data

train_set = torch.from_numpy(x)

train_set = train_set.float()

train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=bs, shuffle=True)

#print(train_set.shape)

#%% Define Network


class Generator(nn.Module):
    def __init__(self, g_input_dim, g_output_dim):
        super(Generator, self).__init__()       
        self.fc1 = nn.Linear(g_input_dim, g_input_dim*2)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features*2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features*2)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)
    
    # forward method
    def forward(self, x): 
        x = F.leaky_relu(self.fc1(x), 0.1) # leaky relu, with slope angle 
        x = F.leaky_relu(self.fc2(x), 0.1) 
        x = F.leaky_relu(self.fc3(x), 0.1)
        #return torch.tanh(self.fc4(x))
        return self.fc4(x) 
    
class Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, d_input_dim *4)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)  # output dim = 1 for binary classification 
    
    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        return torch.sigmoid(self.fc4(x))  # sigmoid for probaility 
    
    
# build network
G = Generator(g_input_dim = z_dim, g_output_dim = nt).to(device)
D = Discriminator(nt).to(device)

# set optimizer 
G_optimizer = optim.Adam(G.parameters(), lr = lr)
D_optimizer = optim.Adam(D.parameters(), lr = lr)


#%% Define training loops 
# 
def G_train(x):
    #=======================Train the generator=======================#
    G.zero_grad()

    z = Variable(torch.randn(bs, z_dim).to(device))
    y = Variable(torch.ones(bs, 1).to(device))

    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()
        
    return G_loss.data.item()

def D_train(x):
    #=======================Train the discriminator=======================#
    D.zero_grad()
    
    # train discriminator on real data
    
    x_real, y_real = x, torch.ones(bs,1)

    x_real, y_real = Variable(x_real.to(device)), Variable(y_real.to(device))

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # train discriminator on fake
    z = Variable(torch.randn(bs, z_dim).to(device))
    x_fake, y_fake = G(z), Variable(torch.zeros(bs, 1).to(device))
    

    D_output = D(x_fake)
    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()
        
    return  D_loss.data.item()


#%% 
for epoch in range(1, n_epoch+1):           
    D_losses, G_losses = [], []
    for batch_idx,x in enumerate(train_loader):
     
        D_losses.append(D_train(x))
        G_losses.append(G_train(x))

    print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
            (epoch), n_epoch, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))
                                                                                                                    
                                                                                                                    
#%% Generate sample 

with torch.no_grad():
    
    fig, ax = plt.subplots(2,4)
    fig.set_figheight(10)
    fig.set_figwidth(20)
    #fig.tight_layout()
    
    
    c = 1
    for i in range(0,2):
        for j in range(0,4):
            z = Variable(torch.randn(z_dim).to(device))
            generated = G(z)
            y = generated.cpu().detach().numpy()
            ax[i,j].plot(t,y)
            ax[i,j].set_title('Sample ' + str(c))
            c+= 1
    
    #plt.show()
    fig.suptitle('Generated Samples number of epochs '+ str(n_epoch),fontsize="x-large")
    plt.savefig('generated_sample.png')
         
                                                                                                                        
# %%
