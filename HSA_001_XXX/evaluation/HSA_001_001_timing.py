import numpy as np
import matplotlib.pyplot as plt
from ICIW_Plots import make_square_ax

plt.style.use("ICIWstyle")

with np.load("HSA_001_XXX/timing/HSA_time_NN_cpu.npz") as data:
    ns_NN_cpu = data["ns"]
    ts_NN_cpu = data["ts"] / 10
with np.load("HSA_001_XXX/timing/HSA_time_EQC.npz") as data:
    ns = data["ns"]
    ts = data["ts"] / 10
with np.load("HSA_001_XXX/timing/HSA_time_NN_gpu_1E2.npz") as data:
    ns_NN_1e2 = data["ns"]
    ts_NN_1e2 = data["ts"] / 10
with np.load("HSA_001_XXX/timing/HSA_time_NN_gpu_1E3.npz") as data:
    ns_NN_1e3 = data["ns"]
    ts_NN_1e3 = data["ts"] / 10
with np.load("HSA_001_XXX/timing/HSA_time_NN_gpu_1E4.npz") as data:
    ns_NN_1e4 = data["ns"]
    ts_NN_1e4 = data["ts"] / 10
with np.load("HSA_001_XXX/timing/HSA_time_NN_gpu_1E5.npz") as data:
    ns_NN_1e5 = data["ns"]
    ts_NN_1e5 = data["ts"] / 10

print(ts)
cm2inch = 1 / 2.54

pos_params = {
    "left_h": 1.65 * cm2inch,
    "bottom_v": 1.1 * cm2inch,
    "ax_width": 5 * cm2inch,
}

fig = plt.figure(figsize=(7 * cm2inch, 9 * cm2inch))
ax = make_square_ax(fig, **pos_params)

ax.set(
    xlabel="n",
    ylabel="$t$ / s",
    xscale="log",
    yscale="log",
)

ax.plot(ns, ts, "--o", label="Shomate")
ax.plot(ns_NN_cpu, ts_NN_cpu, "--o", label="aNN (cpu)")
ax.plot(ns_NN_1e2, ts_NN_1e2, "--o", label="aNN (gpu, 1e2 samples)")
ax.plot(ns_NN_1e3, ts_NN_1e3, "--o", label="aNN (gpu, 1e3 samples)")
# ax.plot(ns_NN_1e4, ts_NN_1e4, "--o", label="aNN (cpu)")
# ax.plot(ns_NN_1e5, ts_NN_1e5, "--o", label="aNN (cpu)")
plt.legend(frameon=True, bbox_to_anchor=(0.4, 1), loc="lower center")

plt.show()
