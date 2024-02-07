# Importe / Bibliotheken
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import root
import matplotlib.pyplot as plt
from scipy.constants import R
import timeit
import sys

sys.path.append(str(Path.cwd() / "lib_nets"))
print(sys.path)


from lib_nets.Nets.EQ_Net_A import NeuralNetwork
from lib_nets.GGW_calc.GGW import GGW
from ICIW_Plots import make_square_ax


# Parameter
num = 10_000_000  # Anzahl der Werte im Vektor

np.random.seed(42)

T = np.random.uniform(408.15, 1273.15, num)  # K Temperatur
p = np.random.uniform(1, 500, num)  # bar Druck

T = torch.tensor(T)
p = torch.tensor(p)

# Stofffmengen zu Reaktionsbeginn
n_ges_0 = 1  # mol Gesamtstoffmenge zum Reaktionsbeginn
x_0 = np.random.dirichlet((1, 1, 1), num)  # 1 Stoffmengenanteile zu Reaktionsbeginn
n_H2_0 = x_0[:, 0] * n_ges_0  # mol Stoffmenge H2 Start
n_N2_0 = x_0[:, 1] * n_ges_0  # mol Stoffmenge N2 Start
n_NH3_0 = x_0[:, 2] * n_ges_0  # mol Stoffmenge NH3 Start

n_H2_0 = torch.tensor(n_H2_0)
n_N2_0 = torch.tensor(n_N2_0)
n_NH3_0 = torch.tensor(n_NH3_0)

X = torch.stack((T, p, n_H2_0, n_N2_0, n_NH3_0), dim=1)

##load the network
torch.set_default_dtype(torch.float64)

net_file = 'C:/Users/TheresaKunz/Python/AG_Güttel_GIT/sr-03-23/models/torch/NH3_net_uniform.pt'
# Path.cwd() / "sr-03-23" / "models" / "torch" / "NH3_net_uniform.pt"

# load model
net = NeuralNetwork(
    5,
    200,
    200,
    200,
    2,
)
net.load_state_dict(torch.load(net_file))
net.eval()


def calc_data(n):
    # Aufruf der GGW-Funktion und Berechnung der Stoffmengen im GGW
    X_ = X[:n]
    # print(X_.shape)
    net(X_)


def main():
    calc_data(10)
    print("calced data")

    ns = np.logspace(1, 6, 6, dtype=int, base=10, endpoint=True)
    n_t = np.sum(ns)
    ts = np.empty_like(ns, dtype=float)
    print(ns)
    for i, n in enumerate(ns):
        cmd = f"calc_data({n})"
        t = timeit.timeit(
            cmd,
            setup="from __main__ import calc_data",
            number=10,
        )
        print(f"{n:<10}:{t:.5f}")
        v = t / n
        n_t = n_t - n
        print(f"\tit/s: {1/v:.3f}")
        eta = 10 * n_t * v
        print(f"\tETA: {eta:.2f}s")
        ts[i] = t

    np.savez("C:/Users/TheresaKunz/Python/AG_Güttel_GIT/sr-03-23/HSA_001_XXX/timing/time_NN_cpu.npz", ns=ns, ts=ts)


if __name__ == "__main__":
    main()
