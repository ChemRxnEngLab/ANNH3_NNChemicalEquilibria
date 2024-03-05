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

data_file = Path.cwd() / "HSA_001_XXX" / "data" / "eq_dataset_loguniform.npz"
net_file = Path.cwd() / "models" / "torch" / "NH3_net_loguniform.pt"

# read in test data
data = np.load(data_file)
T = torch.tensor(data["T"], dtype=torch.float64)[::1000]
p = torch.tensor(data["p"], dtype=torch.float64)[::1000]
x_0 = torch.tensor(data["x_0"], dtype=torch.float64)[::1000, :]
x_eq = torch.tensor(data["x"], dtype=torch.float64)[::1000, :]

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

for name, param in net.named_parameters():
    print(f"name:     {name}")
    print(f"shape:    {param.shape}")
    print(f"req_grad: {param.requires_grad}")
    print("data:")
    print(param.data)

y_pred = net(X).detach()
print(y_pred[0])

### Plotting

fig = plt.figure(figsize=figsize)
ax = make_square_ax(fig, **pos_params)
bounds = (0, 1)

# # Reset the limits
# ax[0] = plt.gca()
ax.set_xlim(bounds)
ax.set_ylim(bounds)
# Ensure the aspect ratio is square
ax.set_aspect("equal", adjustable="box")


ax.scatter(
    x_eq[:, 0],
    y_pred[:, 0],
    # ".",
    label=r"$x\mathregular{_{H_2}}$",
    s=2,
)
ax.scatter(
    x_eq[:, 2],
    y_pred[:, 1],
    label=r"$x\mathregular{_{NH_3}}$",
    s=2,
)

# Error Lines
ax.plot(
    [0, 1],
    [0, 1],
    "-",
    color="k",
    lw=1,
    # transform=ax.transAxes,
    zorder=1,
)
ax.plot(
    [bounds[0], bounds[1]],
    [bounds[0] * 1.1, bounds[1] * 1.1],
    "--",
    color="k",
    lw=0.7,
)
ax.plot(
    [bounds[0], bounds[1]],
    [bounds[0] * 0.9, bounds[1] * 0.9],
    "--",
    color="k",
    lw=0.7,
)

ax.text(0.45, 0.75, "+10%", fontsize=10)
ax.text(0.7, 0.55, "-10%", fontsize=10)
ax.set(
    xlabel="$x_i$ / 1",
    ylabel=r"$x_i\mathregular{_{,pred}}$ / 1",
    xticks=[0, 0.25, 0.5, 0.75, 1],
    yticks=[0, 0.25, 0.5, 0.75, 1],
)
# ax.tick_params(direction="in")  # , length = 20, width = 3)
# ax.set_title("A", loc="left")
ax.legend(frameon=True)
# ax[0].legend(['$\\mathregular{R^2}$ = ', r2(xi_real,xi_pred)], markerscale=0)
x1, x2, y1, y2 = 0, 0.05, 0, 0.05  # subregion of origanal image
axin0 = ax.inset_axes(
    (0.56, 0.04, 0.4, 0.4),
    xlim=(x1, x2),
    ylim=(y1, y2),
    xticks=[0, 0.025, 0.05],
    yticks=[0, 0.025, 0.05],
    xticklabels=[],
    yticklabels=[],
)


axin0.scatter(x_eq[:, 0], y_pred[:, 0], s=2)
axin0.scatter(x_eq[:, 2], y_pred[:, 1], s=2)
# error lines
axin0.plot(
    [0, 1],
    [0, 1],
    "-",
    color="k",
    lw=1,
    zorder=1,
)
axin0.plot(
    [bounds[0], bounds[1]],
    [bounds[0] * 1.1, bounds[1] * 1.1],
    "k--",
    lw=0.7,
)
axin0.plot(
    [bounds[0], bounds[1]],
    [bounds[0] * 0.9, bounds[1] * 0.9],
    "k--",
    lw=0.7,
)

rect = (x1, y1, x2 - x1, y2 - y1)
box2 = ax.indicate_inset(rect, edgecolor="black", alpha=1, lw=0.7)

cp1 = ConnectionPatch(
    xyA=(0.05, 0.0),
    xyB=(0, 0),
    axesA=ax,
    axesB=axin0,
    coordsA="data",
    coordsB="axes fraction",
    lw=0.7,
    ls=":",
    zorder=100,
)
cp2 = ConnectionPatch(
    xyA=(0.05, 0.05),
    xyB=(0, 1),
    axesA=ax,
    axesB=axin0,
    coordsA="data",
    coordsB="axes fraction",
    lw=0.7,
    ls=":",
    zorder=100,
)

ax.add_patch(cp1)
ax.add_patch(cp2)

# plt.legend()
# fig.suptitle("Parity Plot")
plt.tight_layout()
plt.savefig(Path.cwd() / "figures" / "HSA_001_001_parity_loguniform.png", dpi=300)
plt.show()
