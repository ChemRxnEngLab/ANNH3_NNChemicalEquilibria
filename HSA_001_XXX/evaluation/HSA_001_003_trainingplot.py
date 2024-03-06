import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from ICIW_Plots import make_square_ax


plt.style.use("ICIWstyle")

cm2inch = 1 / 2.54

figsize = (7 * cm2inch, 7 * cm2inch)
pos_params = {
    "left_h": 1.7 * cm2inch,
    "bottom_v": 1.05 * cm2inch,
    "ax_width": 5 * cm2inch,
}

losses = np.loadtxt(
    "HSA_001_XXX\evaluation\HSA_001_003.csv",
    delimiter=",",
    skiprows=(1),
    usecols=(0, 4, 7, 13, 16),
)

colors = [
    # "rebeccapurple",
    "teal",
    "orange",
    "limegreen",
    "crimson",
]

fig = plt.figure(figsize=figsize)
ax = make_square_ax(fig, **pos_params)

ax.plot(
    losses[::10, 0],
    losses[::10, 1],
    label="",
    color=colors[0],
)
ax.plot(
    losses[::10, 0],
    losses[::10, 2],
    label="Validation",
    color=colors[0],
    linestyle="--",
)
ax.plot(
    losses[::10, 0],
    losses[::10, 3],
    label="",
    color=colors[3],
)
ax.plot(
    losses[::10, 0],
    losses[::10, 4],
    label="Validation",
    color=colors[3],
    linestyle="--",
)

ax.set_xlabel("epoch")
ax.set_ylabel("loss")

ax.set(
    yscale="log",
)

legend_handles = [
    Line2D([0, 0], [0, 0], linestyle="-", color="k", label="training"),
    Line2D([0, 0], [0, 0], linestyle="--", color="k", label="validation"),
    Line2D([0, 0], [0, 0], linestyle="-", color=colors[0], label="uniform"),
    Line2D([0, 0], [0, 0], linestyle="-", color=colors[3], label="loguniform"),
]

plt.legend(handles=legend_handles, frameon=True)
plt.savefig("figures/training_003.png", dpi=300)

plt.show()
