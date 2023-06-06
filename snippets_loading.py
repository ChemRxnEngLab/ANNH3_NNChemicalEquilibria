import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
from pathlib import Path

#this file does not exist
data_file = Path.cwd() / 'data' / 'ER_003.npz'

res = np.load(data_file)

# when saving the data was saved as X, T and y
# X,T are model inputs and y is the target
# port to torch tensors so pytorch likes it

X = torch.tensor(res['X'])
T = torch.tensor(res['T']).reshape(-1,1)
y = torch.tensor(res['y'])

# group all the tensors in one DataSet
dataset=TensorDataset(X, T, y)

# split it into training and test datasets
train_dataset, test_dataset = random_split(dataset, 
                                           [int(0.8*len(dataset)), int(0.2*len(dataset))], # splitting 80/20
                                           generator=torch.Generator().manual_seed(42)) # setting seed for reproducibility

# convert training and test datasets into dataloaders for later data handling
train_dataloader=DataLoader(train_dataset,batch_size=32,shuffle=True)
test_dataloader=DataLoader(test_dataset,batch_size=32,shuffle=False)