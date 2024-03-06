import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import loguniform
from matplotlib.patches import ConnectionPatch
import sys
from matplotlib import ticker, patches

sys.path.append(str(Path.cwd() / "lib_nets"))
print(sys.path)


from Nets.EQ_Net_A import NeuralNetwork
from GGW_calc.GGW import GGW
from ICIW_Plots import make_square_ax

plt.style.use("ICIWstyle")

cm2inch = 1 / 2.54

figsize = (7 * cm2inch, 7 * cm2inch)

pos_params = {
    "left_h": 1.7 * cm2inch,
    "bottom_v": 1.1 * cm2inch,
    "ax_width": 4.8 * cm2inch,
}

torch.set_default_dtype(torch.float64)

uniform_data_file = Path.cwd() / "HSA_001_XXX" / "data" / "eq_dataset_loguniform.npz"
loguniform_data_file = Path.cwd() / "HSA_001_XXX" / "data" / "eq_dataset_loguniform.npz"
net_file = Path.cwd() / "models" / "torch" / "NH3_net_003_loguniform.pt"

# load model
net = NeuralNetwork(
    5,
    200,
    200,
    200,
    2,
)
net.load_state_dict(torch.load(net_file))

# generate plotting data
data = np.load(uniform_data_file)
num = 100
T = np.linspace(300, 1273.15, num, endpoint=True)  # K temperature
p = np.linspace(1, 700, num, endpoint=True)  # bar pressure
# p = loguniform.rvs(1, 100, size=num)  # bar pressure
Tg, pg = np.meshgrid(T, p)
xg_N2 = np.empty_like(Tg)
xg_H2 = np.empty_like(Tg)
xg_NH3 = np.empty_like(Tg)
xg_net = np.empty_like(Tg)
n_ges_0 = 1  # mol Gesamtstoffmenge Start
x_0 = np.array([0.75, 0.25, 0])
n_0 = n_ges_0 * x_0

for i in range(0, len(p)):
    for j in range(0, len(T)):
        xi, _, _ = GGW(T[j], p[i], n_0)

        n_eq = xi * np.array([-3, -1, 2]) + n_0
        n_ges = n_eq.sum()  # mol Gesamtstoffmenge Gleichgewicht
        xg_H2[i, j], xg_N2[i, j], xg_NH3[i, j] = (
            n_eq / n_ges
        )  # 1 Stoffmengenanteile im Gleichgewicht

        xg_net[i, j] = (
            net(torch.tensor([T[j], p[i], x_0[0], x_0[1], x_0[2]]))
            .squeeze()
            .detach()
            .numpy()
        )[1]

MAE = np.abs(xg_net - xg_NH3) / xg_NH3

fig, ax = plt.subplots()
pcm = ax.contourf(
    pg,
    Tg,
    MAE,
    locator=ticker.LogLocator(),
    levels=np.logspace(-2, 1, 4),
    cmap="Blues",
    extend="both",
)
cl = ax.contour(pcm, levels=np.array([0.1]), colors="k")
ax.clabel(cl, cl.levels, inline=True, fontsize=10)
trained_range = patches.Rectangle(
    (1, 408.15),
    99,
    1273.15 - 408.15,
    linewidth=2,
    linestyle="--",
    edgecolor="k",
    facecolor="none",
)
ax.add_patch(trained_range)
ax.legend()
ax.set(
    xlabel="p / bar",
    ylabel="T / K",
)
fig.colorbar(pcm)
cbar = fig.colorbar(pcm)
cbar.set_label("MAE")

ax.tick_params(direction="out")

plt.grid()

plt.savefig(Path.cwd() / "figures" / "HSA_001_003_parity_loguniform_2D.png", dpi=300)
plt.show()
exit()
