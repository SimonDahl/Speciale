
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

NN=10
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.regressor = nn.Sequential(nn.Linear(1, NN),
                                       nn.Sigmoid(),
                                       nn.Linear(NN, NN),
                                       nn.Sigmoid(),
                                       nn.Linear(NN, NN),
                                       nn.Sigmoid(),
                                       nn.Linear(NN, NN),
                                       nn.Sigmoid(),
                                       nn.Linear(NN, 1))
    def forward(self, x):
        output = self.regressor(x)
        return output

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


## Hyperparameters
LEARNING_RATE = 1e-2
WEIGHT_DECAY = 0
BETA = 1e6
MU = -1;
BETA_LIM = BETA
TRAIN_LIM = 2*np.pi
COL_RES = 500
EPOCHS = 1
#Boundary Conditions
t_bc = np.array([[0]])
x_bc = np.array([[1]])

# Points and boundary vs ODE weight
col_points = int(TRAIN_LIM*COL_RES)
boundary_points = len(x_bc)

F_WEIGHT = 1 #Physics Weight
B_WEIGHT = 1/(10*boundary_points+col_points) #Boundary Weight

# Create net, assign to device and use initialisation
net = Net()
net = net.to(device)
net.apply(init_weights)

# Define loss and optimizer
criterion = torch.nn.MSELoss() # Mean squared error
optimizer = torch.optim.Adam(net.parameters(),lr = LEARNING_RATE)

## PDE as loss function
def f(t,mu,net):
    x = net(t)
    print(t)
    print(t.shape)
    x_t = torch.autograd.grad(x.sum(), t, create_graph=True)[0]
    x_tt = torch.autograd.grad(x.sum(), t, create_graph=True)[0]
    # Test Equation
    ode = mu*x-x_t
    return ode

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
    pt_t_bc = Variable(torch.from_numpy(t_bc).float(), requires_grad=False).to(device)
    pt_x_bc = Variable(torch.from_numpy(x_bc).float(), requires_grad=False).to(device)
    
    print(pt_t_bc)
    print(pt_t_bc.shape)
    
    net_bc_out = net(pt_t_bc) # output of u(x,t)
    
    
    mse_u = criterion(input = net_bc_out, target = pt_x_bc) # Boundary loss

    # Loss based on PDE
    t_collocation = np.random.uniform(low=0.0, high=TRAIN_LIM, size=(col_points,1))
    all_zeros = np.zeros((col_points,1))    
    pt_t_collocation = Variable(torch.from_numpy(t_collocation).float(), requires_grad=True).to(device)
    pt_all_zeros = Variable(torch.from_numpy(all_zeros).float(), requires_grad=False).to(device)
    ode = f(pt_t_collocation,MU,net) # output of f(x,t)
    mse_f = criterion(input = ode, target = pt_all_zeros) #ODE Loss
    
    # Combining the loss functions
    loss,epochBeta = lossCalc(mse_u,mse_f,boundary_points,col_points,F_WEIGHT,B_WEIGHT)
    #Gradients
    loss.backward() 
    #Step Optimizer
    optimizer.step() 
    #Display loss during training
    with torch.autograd.no_grad():
        if epoch%10 == 0:
            print('Epoch:',epoch,"Traning Loss:",loss.data,'epochBeta:',epochBeta)
            print('Boundary Loss:',mse_u/boundary_points,'ODE Loss: ',mse_f/col_points)
        


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

ode1_residual = f(T_test,MU,net)
ode1_residual = ode1_residual.cpu().detach().numpy()

plt.figure()
plt.scatter(T_plot,x1_plot,label = 'X1')
plt.scatter(T_plot,np.exp(-T_plot),label = 'Exact Solution')
plt.legend()
plt.show()

plt.figure()
plt.title('Residual plots of ODE1')
plt.scatter(T_plot,ode1_residual)
plt.show()