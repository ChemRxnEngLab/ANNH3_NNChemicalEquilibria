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

train_losses = np.loadtxt(
    "HSA_001_XXX\evaluation\HSA_001_003_train.csv",
    delimiter=",",
    skiprows=(1),
    usecols=(0, 4, 7, 13, 16),
)

test_losses = np.loadtxt(
    "HSA_001_XXX\evaluation\HSA_001_003_test.csv",
    delimiter=",",
    skiprows=(1),
    usecols=(0, 1, 2),
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
    train_losses[::10, 0],
    train_losses[::10, 1],
    label="",
    color=colors[0],
)
ax.plot(
    train_losses[::10, 0],
    train_losses[::10, 2],
    label="Validation",
    color=colors[0],
    linestyle="--",
)
ax.plot(
    train_losses[::10, 0],
    train_losses[::10, 3],
    label="",
    color=colors[3],
)
ax.plot(
    train_losses[::10, 0],
    train_losses[::10, 4],
    label="Validation",
    color=colors[3],
    linestyle="--",
)

ax.plot(
    test_losses[0],
    test_losses[1],
    "*",
    label="",
    color=colors[0],
    markerfacecolor="none",
    markersize=8,
)
ax.plot(
    test_losses[0],
    test_losses[2],
    "*",
    label="",
    color=colors[3],
    markerfacecolor="none",
    markersize=8,
)

ax.set_xlabel("epoch")
ax.set_ylabel("MSE / 1")

ax.set(
    yscale="log",
)

legend_handles = [
    Line2D([0, 0], [0, 0], linestyle="-", color="k", label="training"),
    Line2D([0, 0], [0, 0], linestyle="--", color="k", label="validation"),
    Line2D(
        [0, 0],
        [0, 0],
        linestyle="none",
        marker="*",
        color="k",
        label="test",
        markerfacecolor="none",
        markersize=8,
    ),
    Line2D([0, 0], [0, 0], linestyle="-", color=colors[0], label="SU"),
    Line2D([0, 0], [0, 0], linestyle="-", color=colors[3], label="SL"),
]

plt.legend(handles=legend_handles, frameon=True)
plt.savefig("figures/training_003.png", dpi=300)

plt.show()
