import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import loguniform
from matplotlib.patches import ConnectionPatch, Rectangle
import sys
from matplotlib import ticker

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
net_file = Path.cwd() / "models" / "torch" / "NH3_net_loguniform.pt"

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
xg_net_H2 = np.empty_like(Tg)
xg_net_NH3 = np.empty_like(Tg)
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

        xg_net_H2[i, j], xg_net_NH3[i, j] = (
            net(torch.tensor([T[j], p[i], x_0[0], x_0[1], x_0[2]]))
            .squeeze()
            .detach()
            .numpy()
        )

MAE_NH3 = np.abs(xg_net_NH3 - xg_NH3) / xg_NH3
MAE_H2 = np.abs(xg_net_H2 - xg_H2) / xg_H2

### NH3

fig, ax = plt.subplots()
pcm = ax.contourf(
    pg,
    Tg,
    MAE_NH3,
    locator=ticker.LogLocator(),
    levels=np.logspace(-2, 1, 4),
    cmap="Blues",
    extend="both",
)
cl = ax.contour(pcm, levels=np.array([0.1]), colors="k")
ax.clabel(cl, cl.levels, inline=True, fontsize=10)
trained_range = Rectangle(
    (1, 408.15),
    499,
    1273.15 - 408.15,
    linewidth=2,
    edgecolor="k",
    facecolor="none",
    linestyle="--",
)
ax.text(75, 350, "trained range", fontsize=12)

ax.add_patch(trained_range)
ax.legend()
ax.set(
    xlabel="p / bar",
    ylabel="T / K",
)
cbar = fig.colorbar(pcm)
cbar.set_label("MRE / 1")

ax.tick_params(direction="out")
### inset axes
in_ax = ax.inset_axes(
    [0.22, 0.22, 0.47, 0.47],
    xlim=(1, 50),
    ylim=(800, 1273.15),
)
in_ax.tick_params(
    direction="out",
    # colors="lightgray",
)
ax.indicate_inset_zoom(in_ax, edgecolor="black")
pcm_2 = in_ax.contourf(
    pg,
    Tg,
    MAE_NH3,
    locator=ticker.LogLocator(),
    levels=np.logspace(-2, 1, 4),
    cmap="Blues",
    extend="both",
)
cl_2 = in_ax.contour(pcm_2, levels=np.array([0.1]), colors="k")
in_ax.clabel(cl_2, cl_2.levels, inline=True, fontsize=10)

plt.grid()

plt.savefig(
    Path.cwd() / "figures" / "HSA_001_001_parity_loguniform_NH3_2D.png", dpi=300
)

### H2

fig, ax = plt.subplots()
pcm = ax.contourf(
    pg,
    Tg,
    MAE_H2,
    locator=ticker.LogLocator(),
    levels=np.logspace(-2, 1, 4),
    cmap="Blues",
    extend="both",
)
cl = ax.contour(pcm, levels=np.array([0.1]), colors="k")
ax.clabel(cl, cl.levels, inline=True, fontsize=10)
trained_range = Rectangle(
    (1, 408.15),
    499,
    1273.15 - 408.15,
    linewidth=2,
    edgecolor="k",
    facecolor="none",
    linestyle="--",
)
ax.text(75, 350, "trained range", fontsize=12)

ax.add_patch(trained_range)
ax.legend()
ax.set(
    xlabel="p / bar",
    ylabel="T / K",
)
cbar = fig.colorbar(pcm)
cbar.set_label("MRE / 1")

ax.tick_params(direction="out")
plt.savefig(Path.cwd() / "figures" / "HSA_001_001_parity_loguniform_H2_2D.png", dpi=300)

plt.show()
