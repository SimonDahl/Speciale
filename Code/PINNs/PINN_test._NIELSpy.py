
"""
PINN  Implementation of The test equation
"""
import torch
import torch.nn as nn
from torch.nn.functional import relu
from torch.autograd import Variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import numpy as np


#class Net(nn.Module):
#    def __init__(self):
#        super(Net, self).__init__()
#        self.hidden_layer1 = nn.Linear(1,1024)
#        self.hidden_layer2 = nn.Linear(1024,1024)
#        self.output_layer = nn.Linear(1024,1)
#
#    def forward(self,x):
#        inputs = x # combined two arrays of 1 columns each to one array of 2 columns
#        layer1_out = relu(self.hidden_layer1(inputs))
#        layer2_out = relu(self.hidden_layer2(layer1_out))
#        output = self.output_layer(layer2_out) ## For regression, no activation is used in output layer
#        return output
#
#    def predict(self, X):
#            X = torch.Tensor(X)
#            return self(X).detach().numpy().squeeze()




NN=50
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.regressor = nn.Sequential(nn.Linear(1, NN),
                                       nn.Tanh(),
                                       nn.Linear(NN, NN),
                                       nn.Tanh(),
                                       nn.Linear(NN, NN),
                                       nn.Tanh(),
                                       nn.Linear(NN, 2))
    def forward(self, x):
        output = self.regressor(x)
        return output

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)


## Hyperparameters
LEARNING_RATE = 1e-2
WEIGHT_DECAY = 0
BETA = 1e6
MU = -1;
BETA_LIM = BETA
TRAIN_LIM = 3*np.pi
COL_RES = 1000
EPOCHS = 1000

#Boundary Conditions
t_bc = np.array([[0]])
x1_bc = np.array([[1]])
x2_bc = np.array([[5]])

# Points and boundary vs ODE weight
col_points = int(TRAIN_LIM*COL_RES)


M=5
K=2
C=0.3


# Create net, assign to device and use initialisation
net = Net()
net = net.to(device)
net.apply(init_weights)

# Define loss and optimizer
criterion = torch.nn.MSELoss() # Mean squared error
optimizer = torch.optim.Adam(net.parameters(),lr = LEARNING_RATE)


def net_u(x):        
    u = net(x)    
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0] # automatic differentiation 
    return u, u_x

## PDE as loss function
def f(t,mu,net,m,c,k):
    x1,x2 = net(t)
    x1_t = torch.autograd.grad(x1.sum(), t, create_graph=True)[0]
    x2_t = torch.autograd.grad(x2.sum(), t, create_graph=True)[0]
    # Test Equation
    ode1 = x1_t-x2
    ode2 = c*x2+k*x1-m*x2_t
    return ode1,ode2

def lossCalc(mse_u,mse_f,bp,cp,f_weight,b_weight,epoch = -1,beta = 1,betaLim = 1):
    # For implementing curriculum learning by varying epoch*beta
    if epoch*beta > betaLim or epoch == -1:
        loss = (b_weight*mse_u)/bp + (f_weight*mse_f/cp)
        epochBeta = betaLim
    else:
        loss = (b_weight*mse_u)/bp + (f_weight*mse_f/cp)*epoch*beta
        epochBeta = epoch*beta
    
    return loss,epochBeta


for epoch in range(EPOCHS):
    optimizer.zero_grad() # to make the gradients zero
    
    # Loss based on boundary conditions
    pt_t_bc = Variable(torch.from_numpy(t_bc).float(), requires_grad=True).to(device)
    pt_x1_bc = Variable(torch.from_numpy(x1_bc).float(), requires_grad=True).to(device)
    pt_x2_bc  =Variable(torch.from_numpy(x2_bc).float(), requires_grad=True).to(device)
    
    net_bc_x1,net_bc_x2 = net(pt_t_bc) # output of u(x,t)

    mse_u = criterion(input = net_bc_x1, target = pt_x1_bc)+ criterion(input = net_bc_x2, target = pt_x2_bc)# Boundary loss

    # Loss based on PDE
    t_collocation = np.random.uniform(low=0.0, high=TRAIN_LIM, size=(col_points,1))
    all_zeros = np.zeros((col_points,1))    
    pt_t_collocation = Variable(torch.from_numpy(t_collocation).float(), requires_grad=True).to(device)
    pt_all_zeros = Variable(torch.from_numpy(all_zeros).float(), requires_grad=False).to(device)
    ode = f(pt_t_collocation,MU,net,M,C,K) # output of f(x,t)
    mse_f = criterion(input = ode, target = pt_all_zeros) #ODE Loss
    
    # Combining the loss functions
    loss = mse_f+mse_u
    #Gradients
    loss.backward() 
    #Step Optimizer
    optimizer.step() 
    #Display loss during training
    with torch.autograd.no_grad():
        if epoch%10 == 0:
            print('Epoch:',epoch,"Traning Loss:",loss.data)
            print('Boundary Loss:',mse_u,'ODE Loss: ',mse_f)
        



import matplotlib.pyplot as plt
import numpy as np

n = 1000
T_test = torch.linspace(0,TRAIN_LIM,n,requires_grad=True).to(device)
T_test = T_test.reshape(n,1)

score = net(T_test) 

x1_plot = score.cpu().detach().numpy()

T_plot = torch.linspace(0,TRAIN_LIM,n,requires_grad=False)
T_plot = T_test.reshape(n,1)
T_plot = T_plot.cpu().detach().numpy()

ode1_residual = f(T_test,MU,net,M,C,K)
ode1_residual = ode1_residual.cpu().detach().numpy()




#x0 = 5
#dx/dt = 0
#m = 1
#c = 1
#k = 2
#Solution
#y = (5*np.sqrt(7)*np.exp(-T_plot/2)*np.sin(np.sqrt(7)*T_plot/2))/7+(5*np.exp(-T_plot/2)*np.cos(np.sqrt(7)*T_plot/2))



#x0 = 5
#dx/dt = 0
#m = 1
#c = 0.3
#k = 2
#Solution
#y = (75*np.sqrt(791)*np.exp(-3*T_plot/20)*np.sin(np.sqrt(791)*T_plot/20))/791+(5*np.exp(-3*T_plot/20)*np.cos(np.sqrt(791)*T_plot/20))


#x0 = 0
#dx/dt = 30
#m = 1
#c = 0.3
#k = 2
#Solution
y = (600*np.sqrt(791)*np.exp(-3*T_plot/20)*np.sin(np.sqrt(791)*T_plot/20))/791


plt.figure()
plt.scatter(T_plot,x1_plot,label = 'X1')
plt.scatter(T_plot,y,label = 'Exact')

plt.legend()

plt.figure()
plt.title('Residual plots of ODE1')
plt.scatter(T_plot,ode1_residual)
