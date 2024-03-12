# import torch
# import torch.nn as nn
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import sys
from matplotlib.lines import Line2D

sys.path.append(str(Path.cwd() / "lib_nets"))
print(Path.cwd())
print(sys.path)


# from Nets.EQ_Net_B import NeuralNetwork
from GGW_calc.GGW import GGW
from ICIW_Plots import make_square_ax

plt.style.use("ICIWstyle")

cm2inch = 1 / 2.54

# figsize = (15 * cm2inch, 7 * cm2inch)
figsize = (10.5 * cm2inch, 5.5 * cm2inch)

pos_params = {
    "left_h": 1.7 * cm2inch,
    "bottom_v": 1.1 * cm2inch,
    "ax_width": 4.8 * cm2inch,
}

# torch.set_default_dtype(torch.float64)
#
# net_file = Path.cwd() / "models" / "torch" / "BatchNorm_NH3_net_LU_001.pt"

# # load model
# net = NeuralNetwork(
#     5,
#     200,
#     200,
#     200,
#     2,
# )
# net.load_state_dict(torch.load(net_file))
#
# for name, param in net.named_parameters():
#     print(f"name:     {name}")
#     print(f"shape:    {param.shape}")
#     print(f"req_grad: {param.requires_grad}")
#     print("data:")
#     print(param.data)


### GGW Verläufe

# Plots
num_plot = 50  # Anzahl der berechneten Punkte
n_ges_0_plot = 1  # mol Gesamtstoffmenge Start
x_0_plot = np.array([0.762, 0.238, 0])
n = n_ges_0_plot * x_0_plot

# Diagramm1: Parameter zur Berechnung von xi über T bei versch. Druecken
T_plot1 = np.linspace(300 + 273.15, 550 + 273.15, num=num_plot)  # K Temperatur
p_plot1 = np.array([10, 30, 50, 100, 300, 600, 1000]) * 1.01325  # bar Druck;

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

        # x_net_plot1[j, i, :] = (
        #     net(
        #         torch.tensor(
        #             [T_plot1[j], p_plot1[i], x_0_plot[0], x_0_plot[1], x_0_plot[2]]
        #         )
        #     )
        #     .squeeze()
        #     .detach()
        #     .numpy()
        # )

# validation data (Larson 1923, doi.org/10.1021/ja01665a017 and Larson 1924, doi.org/10.1021/ja01667a011)

T_larson = (
    np.array([325, 350, 375, 400, 425, 450, 475, 500]) + 273.15
)  # temperature in K
larson = (
    np.array(
        [
            [10.38, 7.35, 5.25, 3.85, 2.80, 2.04, 1.61, 1.20],  # 10 atm, x_NH3 in %
            [np.nan, 17.80, 13.35, 10.09, 7.59, 5.80, 4.53, 3.48],  # 30 atm
            [np.nan, 25.11, 19.44, 15.11, 11.71, 9.17, 7.13, 5.58],  # 50 atm
            [np.nan, np.nan, 30.95, 24.91, 20.23, 16.35, 12.98, 10.40],  # 100 atm
            [np.nan, np.nan, np.nan, np.nan, np.nan, 35.5, 31.0, 26.2],  # 300 atm
            [np.nan, np.nan, np.nan, np.nan, np.nan, 53.6, 47.5, 42.1],  # 600 atm
            [np.nan, np.nan, np.nan, np.nan, np.nan, 69.4, 63.5, np.nan],  # 1000 atm
        ]
    )
    / 100
)

# xi über T bei unterschiedlichen p
fig1, ax1 = plt.subplots()

colors = [
    "rebeccapurple",
    "teal",
    "orange",
    "limegreen",
    "crimson",
    "orangered",
    "magenta",
]

for i in range(0, len(p_plot1)):
    ax1.plot(
        T_plot1,
        x_plot1[:, i, 2],
        "-",
        label=f"$p$ = {p_plot1[i]/1.01325} atm",
        color=colors[i],
    )

for j in range(larson.shape[0]):
    ax1.plot(
        T_larson,
        larson[j, :],
        "P",
        label=f"$p$ = {p_plot1[j]/1.01325} atm (Larson)",
        color=colors[j],
        markeredgecolor="k",
        markersize=7,
    )
    # ax1.plot(
    #     T_plot1,
    #     x_net_plot1[:, i, 1],
    #     "o",
    #     color=colors[i],
    # )

#'o': Punkte;'-': Verbindung mit Linien; '--':gestrichelte Linie...
# Farbe ändern: b blau; r rot; g grün; y yellow; m magenta; c cyan; schwarz k; w weiß
ax1.set(
    xlabel="$\mathit{T}$ / K",
    ylabel=r"$\mathit{x}_{\mathrm{NH}_3}$ / 1",
    ylim=(0, 1),
)  # Beschriftung Achsen; Kursiv durch $$; Index durch _{}
# ax1.tick_params(direction="in", length=20, width=3)
ax1.set(xlim=(T_plot1[0], T_plot1[-1]))

legend_handles = [
    Line2D(
        [0, 0],
        [0, 0],
        color=colors[0],
        lw=2,
        label=f"$p$ = {p_plot1[0]/1.01325} atm",
    ),
    Line2D(
        [0, 0],
        [0, 0],
        color=colors[1],
        lw=2,
        label=f"$p$ = {p_plot1[1]/1.01325} atm",
    ),
    Line2D(
        [0, 0],
        [0, 0],
        color=colors[2],
        lw=2,
        label=f"$p$ = {p_plot1[2]/1.01325} atm",
    ),
    Line2D(
        [0, 0],
        [0, 0],
        color=colors[3],
        lw=2,
        label=f"$p$ = {p_plot1[3]/1.01325} atm",
    ),
    Line2D(
        [0, 0],
        [0, 0],
        color=colors[4],
        lw=2,
        label=f"$p$ = {p_plot1[4]/1.01325} atm",
    ),
    Line2D(
        [0, 0],
        [0, 0],
        color=colors[5],
        lw=2,
        label=f"$p$ = {p_plot1[5]/1.01325} atm",
    ),
    Line2D(
        [0, 0],
        [0, 0],
        color=colors[6],
        lw=2,
        label=f"$p$ = {p_plot1[6]/1.01325} atm",
    ),
    Line2D(
        [0, 0],
        [0, 0],
        linestyle="None",
        marker="P",
        color="gray",
        lw=2,
        label="Larson exp. data",
        markersize=7,
        markeredgecolor="k",
    ),
]

leg1 = ax1.legend(
    handles=legend_handles, frameon=True, bbox_to_anchor=(1, 1), loc="upper left"
)  # Legende anzeigen
# leg1.get_frame().set_edgecolor("k")  # schwarzer Kasten um Legende
# leg1.get_frame().set_linewidth(3)  # Linienstärke Kasten um Legende


plt.tight_layout()
# Anzeigen der Diagramme
plt.savefig("figures/val_001_Shomate_Larson.png", dpi=300, bbox_inches="tight")
plt.show()
