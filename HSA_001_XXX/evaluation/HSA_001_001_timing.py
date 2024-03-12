import numpy as np
import matplotlib.pyplot as plt
from ICIW_Plots import make_square_ax
from pathlib import Path

plt.style.use("ICIWstyle")

base_path = Path.cwd() / "HSA_001_XXX" / "timing"

with np.load(base_path / "HSA_time_NN_cpu.npz") as data:
    ns_NN_cpu = data["ns"]
    ts_NN_cpu = data["ts"] / 10
with np.load(base_path / "HSA_time_EQC.npz") as data:
    ns = data["ns"]
    ts = data["ts"] / 10
with np.load(base_path / "HSA_time_NN_gpu_1E2.npz") as data:
    ns_NN_1e2 = data["ns"]
    ts_NN_1e2 = data["ts"] / 10
with np.load(base_path / "HSA_time_NN_gpu_1E3.npz") as data:
    ns_NN_1e3 = data["ns"]
    ts_NN_1e3 = data["ts"] / 10
with np.load(base_path / "HSA_time_NN_gpu_1E4.npz") as data:
    ns_NN_1e4 = data["ns"]
    ts_NN_1e4 = data["ts"] / 10
with np.load(base_path / "HSA_time_NN_gpu_1E5.npz") as data:
    ns_NN_1e5 = data["ns"]
    ts_NN_1e5 = data["ts"] / 10

print(ts)
cm2inch = 1 / 2.54

pos_params = {
    "left_h": 1.65 * cm2inch,
    "bottom_v": 1.1 * cm2inch,
    "ax_width": 5 * cm2inch,
}

fig = plt.figure(figsize=(8 * cm2inch, 8 * cm2inch))
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
# plt.legend(frameon=True, bbox_to_anchor=(0.4, 1), loc="lower center")

plt.tight_layout()
plt.savefig(Path.cwd() / "figures" / "HSA_001_001_timing.png", dpi=300)
plt.savefig(Path.cwd() / "figures" / "HSA_001_001_timing.svg")

plt.show()

lambda_NN_cpu = ts[1:] / ts_NN_cpu
lambda_NN_1e2 = ts[1:] / ts_NN_1e2
lambda_NN_1e3 = ts[1:] / ts_NN_1e3
lambda_NN_1e4 = ts[1:] / ts_NN_1e4
lambda_NN_1e5 = ts[1:] / ts_NN_1e5

fig = plt.figure(figsize=(7 * cm2inch, 9 * cm2inch))
ax = make_square_ax(fig, **pos_params)

ax.set(
    xlabel="n",
    ylabel="$\Phi$ / 1",
    xscale="log",
    yscale="log",
)

ax.plot(
    ns_NN_cpu,
    lambda_NN_cpu,
    "o",
    label="aNN (cpu)",
)
ax.plot(ns_NN_cpu, lambda_NN_cpu, "--o", label="aNN (cpu)")
ax.plot(ns_NN_1e2, lambda_NN_1e2, "--o", label="aNN (gpu, 1e2 samples)")
ax.plot(ns_NN_1e3, lambda_NN_1e3, "--o", label="aNN (gpu, 1e3 samples)")
# ax.plot(ns_NN_1e4, lambda_NN_1e4, "--o", label="aNN (cpu)")
# ax.plot(ns_NN_1e5, lambda_NN_1e5, "--o", label="aNN (cpu)")

ax.set_ylim(1e-1, 1e2)

plt.legend(frameon=True, loc="lower right")
plt.tight_layout()
plt.savefig(Path.cwd() / "figures" / "HSA_001_001_speedup.png", dpi=300)
plt.savefig(Path.cwd() / "figures" / "HSA_001_001_speedup.svg")
plt.show()
