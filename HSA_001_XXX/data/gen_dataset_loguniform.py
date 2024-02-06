# Imports
from pathlib import Path
import numpy as np
from scipy.stats import loguniform
from lib_nets.GGW_calc.GGW import GGW

# Parameters
num = 1_000_000  # number of training samples

T = np.random.uniform(408.15, 1273.15, num)  # K temperature
p = loguniform.rvs(1, 500, size=num)  # bar pressure
n_ges_0 = 1  # mol total amount at reaction start

# Stofffmengen zu Reaktionsbeginn
x_0 = np.random.dirichlet((1, 1, 1), num)  # 1 molar fraction at initial condition
n_0 = x_0 * n_ges_0  # mol amounts at reaction start
n_H2_0 = x_0[:, 0] * n_ges_0  # mol amount H2 at reaction start
n_N2_0 = x_0[:, 1] * n_ges_0  # mol amount N2 at reaction start
n_NH3_0 = x_0[:, 2] * n_ges_0  # mol amount NH3 at reaction start


data_path = Path.cwd() / "HSA_001_XXX" / "data"
data_file = data_path / "eq_dataset_loguniform.npz"

if not data_path.exists():
    print(f"{data_path} does not exist. Create it!")

if data_file.exists():
    print("orerriding existing data file")

# calculate EQ for each sample
xi = np.zeros_like(n_H2_0)
for i in range(0, len(n_H2_0)):
    if i % 1000 == 0:
        print(f"{i} of {num}")
    xi[i], _, _ = GGW(T[i], p[i], n_0[i, :])

# calculate the EQ-composition
v_H2 = -3
v_N2 = -1
v_NH3 = 2

n_H2 = xi * v_H2 + n_H2_0  # mol amount H2 in Eq
n_N2 = xi * v_N2 + n_N2_0  # mol amount N2 in Eq
n_NH3 = xi * v_NH3 + n_NH3_0  # mol amount NH3 in Eq
n_ges = n_H2 + n_N2 + n_NH3  # mol total amount in Eq
x = (np.array([n_H2, n_N2, n_NH3]) / n_ges).T  # 1 molar fractions in Eq

# save data
np.savez(data_file, T=T, p=p, x_0=x_0, x=x)
