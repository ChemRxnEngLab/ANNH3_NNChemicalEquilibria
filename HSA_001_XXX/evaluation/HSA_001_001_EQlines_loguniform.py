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

("ICIWstyle")

cm2inch = 1 / 2.54

figsize = (15 * cm2inch, 7 * cm2inch)

pos_params = {
    "left_h": 1.7 * cm2inch,
    "bottom_v": 1.1 * cm2inch,
    "ax_width": 4.8 * cm2inch,
}

torch.set_default_dtype(torch.float64)

net_file = Path.cwd() / "models" / "torch" / "NH3_net_LL.pt"

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


### GGW Verläufe

# Plots
num_plot = 50  # Anzahl der berechneten Punkte
n_ges_0_plot = 1  # mol Gesamtstoffmenge Start
x_0_plot = np.array([0.75, 0.25, 0])
n = n_ges_0_plot * x_0_plot

# Diagramm1: Parameter zur Berechnung von xi über T bei versch. Druecken
T_plot1 = np.linspace(300, 1300, num=num_plot)  # K Temperatur
p_plot1 = np.array([1, 20, 100, 500, 700])  # bar Druck;

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
                )
            )
            .squeeze()
            .detach()
            .numpy()
        )


# xi über T bei unterschiedlichen p
fig1, ax1 = plt.subplots(figsize=figsize)

colors = {
    1: "rebeccapurple",
    20: "teal",
    100: "orange",
    200: "limegreen",
    500: "crimson",
    700: "mediumvioletred",
}

for i in range(0, len(p_plot1)):
    ax1.plot(
        T_plot1,
        x_plot1[:, i, 2],
        "-",
        label=f"$p$ = {p_plot1[i]} bar",
        color=colors[p_plot1[i]],
    )
    ax1.plot(
        T_plot1,
        x_net_plot1[:, i, 1],
        "o",
        color=colors[p_plot1[i]],
    )

#'o': Punkte;'-': Verbindung mit Linien; '--':gestrichelte Linie...
# Farbe ändern: b blau; r rot; g grün; y yellow; m magenta; c cyan; schwarz k; w weiß
ax1.set(
    xlabel="$T$ / K",
    ylabel=r"$x_{\mathrm{NH}_3}$ / 1",
    ylim=(0, 1),
)  # Beschriftung Achsen; Kursiv durch $$; Index durch _{}
# ax1.tick_params(direction="in", length=20, width=3)
ax1.set(xlim=(T_plot1[0], T_plot1[-1]))

leg1 = ax1.legend(frameon=True)  # Legende anzeigen
# leg1.get_frame().set_edgecolor("k")  # schwarzer Kasten um Legende
# leg1.get_frame().set_linewidth(3)  # Linienstärke Kasten um Legende

### the trained area
# The trained temperature range
cond_idcs = np.where(np.logical_and(T_plot1 > 408.15, T_plot1 < 1273.15))[0]
print(cond_idcs)
T_trained = T_plot1[cond_idcs]
print(T_trained.shape)
print(T_trained)
fill = ax1.fill_between(
    T_trained,
    x_plot1[cond_idcs, 0, 2],
    x_plot1[cond_idcs, 3, 2],
    color="lightgray",
    alpha=1,
    zorder=1,
)  # , label = 'Haber-Bosch')#,alpha = 0.6)

# Beschriftung Fläche
(x0, y0), (x1, y1) = fill.get_paths()[0].get_extents().get_points()
middle_x = (x0 + x1) / 2
middle_y = (y0 + y1) / 2
ax1.text(
    800,
    (y0 + y1) / 2 + 0.2,
    "Trained\narea",
    ha="center",
    va="bottom",
    fontsize=11,
    color="black",
    zorder=100,
)
con = ConnectionPatch(
    xyA=(700, 0.25),
    coordsA=ax1.transData,
    xyB=(800, (y0 + y1) / 2 + 0.2),
    coordsB=ax1.transData,
    lw=0.7,
    zorder=99,
)
ax1.add_artist(con)


plt.tight_layout()
# Anzeigen der Diagramme
plt.savefig(Path.cwd() / "figures" / "HSA_001_001_EQlines_loguniform.png", dpi=300)
plt.show()
