import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import sys

sys.path.append(str(Path.cwd() / "lib_nets"))
print(sys.path)

from Nets.EQ_Net_A import NeuralNetwork


# torch.set_default_dtype(torch.float64)

data_file = Path.cwd() / "HSA_001_XXX" / "data" / "eq_dataset_uniform.npz"
net_file = Path.cwd() / "models" / "torch" / "NH3_net_uniform.pt"

# read in test data
data = np.load(data_file)
T = torch.tensor(data["T"], dtype=torch.float32)[::1000]
p = torch.tensor(data["p"], dtype=torch.float32)[::1000]
x_0 = torch.tensor(data["x_0"], dtype=torch.float32)[::1000, :]
x_eq = torch.tensor(data["x"], dtype=torch.float32)[::1000, :]

X = torch.cat((T[:, None], p[:, None], x_0), 1)
y = x_eq

# load model
net = NeuralNetwork(
    5,
    200,
    200,
    200,
    2,
)
net.load_state_dict(torch.load(net_file))

torch.onnx.export(
    net,
    X,
    net_file.with_suffix(".onnx"),
)
