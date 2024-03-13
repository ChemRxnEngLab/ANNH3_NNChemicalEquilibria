import sys
from pathlib import Path
import torch

sys.path.append(str(Path.cwd() / "lib_nets"))
print(sys.path)

from Nets.EQ_Net_A import NeuralNetwork

saved_model_file = "models/torch/NH3_net_LL.pt"

# empty (untrained Network)
model = NeuralNetwork.default_config()
model.load_state_dict(torch.load(saved_model_file))

# network instantiated from saved model
model = NeuralNetwork.from_state_dict(saved_model_file)

p = 25e5
T = 500
x_initial = [0.25, 0.75, 0]

x_eq = model(torch.tensor([p, T, *x_initial]))
