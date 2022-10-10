import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import utils
from model import VAE

# setting
datadir = r'C:\Users\Ejer\OneDrive - Danmarks Tekniske Universitet\Dokumenter\GitHub\Speciale\Code\PI_VAE\data'
dataname = 'train_data'
modeldir = './out_pendulum/'


# load data
data_test = np.loadtxt('{}/data_{}.txt'.format(datadir, dataname))

# load ture parameters
params_test = np.loadtxt('{}/true_params_{}.txt'.format(datadir, dataname))

device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set and load model (aux only)
with open('{}/args.json'.format(modeldir), 'r') as f:
    args_tr_dict = json.load(f)
model = VAE(args_tr_dict).to(device)
model.load_state_dict(torch.load('{}/model.pt'.format(modeldir), map_location=device))
model.eval()

dim_t_tr = args_tr_dict['dim_t']
dt = args_tr_dict['dt']