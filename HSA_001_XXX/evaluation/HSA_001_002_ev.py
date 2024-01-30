# %%
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import sys

sys.path.append(str(Path.cwd() / "lib_nets"))


from Nets.EQ_Net_B import NeuralNetwork
from GGW_calc.GGW import GGW

torch.set_default_dtype(torch.float64)

data_file = Path.cwd() / "HSA_001_XXX" / "data" / "eq_dataset.npz"
net_file = Path.cwd() / "models" / "torch" / "extended_NH3_net_BatchNorm.pt"

# read in test data
data = np.load(data_file)
T = torch.tensor(data["T"], dtype=torch.float64)
p = torch.tensor(data["p"], dtype=torch.float64)
x_0 = torch.tensor(data["x_0"], dtype=torch.float64)
x_eq = torch.tensor(data["x"], dtype=torch.float64)

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
# %%
### Plotting

fig, ax = plt.subplots(1, 2)  # , figsize =(13*cm,6.5*cm))
bounds = (0, 1)

# # Reset the limits
# ax[0] = plt.gca()
ax[0].set_xlim(bounds)
ax[0].set_ylim(bounds)
# Ensure the aspect ratio is square
ax[0].set_aspect("equal", adjustable="box")


ax[0].scatter(
    x_eq[:, 0],
    y_pred[:, 0],
    label=r"$x\mathregular{_{H_2}}$",
    s=2,
)
ax[0].scatter(
    x_eq[:, 2],
    y_pred[:, 1],
    label=r"$x\mathregular{_{NH_3}}$",
    s=2,
)
ax[0].plot(
    [0, 1],
    [0, 1],
    "-",
    color="crimson",
    lw=1.5,
    transform=ax[0].transAxes,
    zorder=1,
)
ax[0].plot(
    [bounds[0], bounds[1]],
    [bounds[0] * 1.1, bounds[1] * 1.1],
    "k--",
    lw=1,
)  # Error line
ax[0].plot(
    [bounds[0], bounds[1]],
    [bounds[0] * 0.9, bounds[1] * 0.9],
    "k--",
    lw=1,
)  # Error line

ax[0].text(0.45, 0.75, "+10%", fontsize=7)
ax[0].text(0.7, 0.55, "-10%", fontsize=7)
ax[0].set(xlabel="$x_i$ / 1", ylabel=r"$x_i\mathregular{_{,pred}}$ / 1")
ax[0].tick_params(direction="in")  # , length = 20, width = 3)
ax[0].set_title("A", loc="left")
ax[0].legend()
# ax[0].legend(['$\\mathregular{R^2}$ = ', r2(xi_real,xi_pred)], markerscale=0)
x1, x2, y1, y2 = 0, 0.05, 0, 0.05  # subregion of origanal image
axin0 = ax[0].inset_axes(
    [0.56, 0.04, 0.4, 0.4],
    xlim=(x1, x2),
    ylim=(y1, y2),
    xticks=[],
    yticks=[],
    xticklabels=[],
    yticklabels=[],
)

axin0.scatter(x_eq[:, 0], y_pred[:, 0], s=2)
axin0.scatter(x_eq[:, 2], y_pred[:, 1], s=2)
axin0.plot([0, 1], [0, 1], "-", color="crimson", lw=1, zorder=1)
axin0.plot([bounds[0], bounds[1]], [bounds[0] * 1.1, bounds[1] * 1.1], "k--", lw=1)
axin0.plot([bounds[0], bounds[1]], [bounds[0] * 0.9, bounds[1] * 0.9], "k--", lw=1)

rect = (x1, y1, x2 - x1, y2 - y1)
box2 = ax[0].indicate_inset(rect, edgecolor="black", alpha=1, lw=0.7)

cp1 = ConnectionPatch(
    xyA=(0.05, 0.0),
    xyB=(0, 0),
    axesA=ax[0],
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
    axesA=ax[0],
    axesB=axin0,
    coordsA="data",
    coordsB="axes fraction",
    lw=0.7,
    ls=":",
    zorder=100,
)

ax[0].add_patch(cp1)
ax[0].add_patch(cp2)

# plt.legend()
# fig.suptitle("Parity Plot")
plt.tight_layout()
plt.show()

### GGW Verläufe

# Plots
num_plot = 50  # Anzahl der berechneten Punkte
n_ges_0_plot = 1  # mol Gesamtstoffmenge Start
x_0_plot = np.array([0.75, 0.25, 0])
n = n_ges_0_plot * x_0_plot

# Diagramm1: Parameter zur Berechnung von xi über T bei versch. Druecken
T_plot1 = np.linspace(300, 1200, num=num_plot)  # K Temperatur
p_plot1 = np.array([10, 200, 500])  # bar Druck;

# Aufrufen der Funktion zur Berechnung von xi mit Shomate
K_x_plot1 = np.zeros((num_plot, len(p_plot1)))
xi_plot1 = np.zeros((num_plot, len(p_plot1)))
x_plot1 = np.zeros((num_plot, len(p_plot1), 3))
x_net_plot1 = np.zeros((num_plot, len(p_plot1), 2))
for i in range(0, len(p_plot1)):
    for j in range(0, len(T_plot1)):
        xi_plot1[j, i], K_x_plot1[j, i], _ = GGW(T_plot1[j], p_plot1[i], n)

        n_eq = xi_plot1[j, i] * np.array([-3, -1, 2]) + n
        n_ges = n_eq.sum()  # mol Gesamtstoffmenge Gleichgewicht
        x_plot1[j, i, :] = n_eq / n_ges  # 1 Stoffmengenanteile im Gleichgewicht

        x_net_plot1[j, i, :] = (
            net(
                torch.tensor(
                    [T_plot1[j], p_plot1[i], x_0_plot[0], x_0_plot[1], x_0_plot[2]]
                ).reshape(1, -1)
            )
            .squeeze()
            .detach()
            .numpy()
        )

# Diagramme zeichnen
# Allgemeine Formatierung
# plt.rc("font", size=40)  # Schriftgroesse
# plt.rc("lines", linewidth=7)  # Linienstaerke
# plt.rcParams["axes.linewidth"] = 3  # Dicke Rahmenlinie


# xi über T bei unterschiedlichen p
fig1, ax1 = plt.subplots()
plt.grid()
ax2 = ax1.twinx()
ax1.plot(
    T_plot1,
    x_plot1[:, 0, 2],
    "-",
    color="rebeccapurple",
    label="$p$ = 100 bar",
    lw=1.5,
)
# ax2.plot(
#     T_plot1,
#     xi_plot1[:, 0],
#     ":",
#     color="rebeccapurple",
# )
ax1.plot(
    T_plot1,
    x_net_plot1[:, 0, 1],
    "o",
    color="rebeccapurple",
)
ax1.plot(
    T_plot1,
    x_plot1[:, 1, 2],
    "-",
    color="teal",
    label="$p$ = 200 bar",
    lw=1.5,
)
# ax2.plot(
#     T_plot1,
#     xi_plot1[:, 1],
#     ":",
#     color="teal",
# )
ax1.plot(
    T_plot1,
    x_net_plot1[:, 1, 1],
    "o",
    color="teal",
)
ax1.plot(
    T_plot1,
    x_plot1[:, 2, 2],
    "-",
    color="orange",
    label="$p$ = 300 bar",
    lw=1.5,
)
# ax2.plot(
#     T_plot1,
#     xi_plot1[:, 2],
#     ":",
#     color="orange",
# )
ax1.plot(
    T_plot1,
    x_net_plot1[:, 2, 1],
    "o",
    color="orange",
)
#'o': Punkte;'-': Verbindung mit Linien; '--':gestrichelte Linie...
# Farbe ändern: b blau; r rot; g grün; y yellow; m magenta; c cyan; schwarz k; w weiß
ax1.set(
    xlabel="$T$ / K", ylabel=r"$x_{\mathrm{NH}_3}$ / 1"
)  # Beschriftung Achsen; Kursiv durch $$; Index durch _{}
# ax1.tick_params(direction="in", length=20, width=3)
ax1.set(xlim=(T_plot1[0], T_plot1[-1]))

leg1 = ax1.legend()  # Legende anzeigen
# leg1.get_frame().set_edgecolor("k")  # schwarzer Kasten um Legende
# leg1.get_frame().set_linewidth(3)  # Linienstärke Kasten um Legende

plt.tight_layout()
# Anzeigen der Diagramme
plt.show()
