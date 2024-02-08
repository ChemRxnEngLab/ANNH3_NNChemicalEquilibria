# Importe / Bibliotheken
import numpy as np
import timeit
from lib_nets.GGW_calc.GGW import GGW


# Parameter
num = 10_000_000  # Anzahl der Werte im Vektor

np.random.seed(42)

T = np.random.uniform(408.15, 1273.15, num)  # K Temperatur
p = np.random.uniform(1, 500, num)  # bar Druck
n_ges_0 = 1  # mol Gesamtstoffmenge zum Reaktionsbeginn

# Stofffmengen zu Reaktionsbeginn
x_0 = np.random.dirichlet((1, 1, 1), num)  # 1 Stoffmengenanteile zu Reaktionsbeginn
n_0 = x_0 * n_ges_0  # mol Stoffmengen zu Reaktionsbeginn
n_H2_0 = x_0[:, 0] * n_ges_0  # mol Stoffmenge H2 Start
n_N2_0 = x_0[:, 1] * n_ges_0  # mol Stoffmenge N2 Start
n_NH3_0 = x_0[:, 2] * n_ges_0  # mol Stoffmenge NH3 Start


def calc_data(n):
    # Aufruf der GGW-Funktion und Berechnung der Stoffmengen im GGW
    for i in range(0, n):
        GGW(T[i], p[i], n_0[i, :])


def main():
    calc_data(1)
    print("calced data")

    ns = np.logspace(0, 6, 7, dtype=int, base=10, endpoint=True)
    n_t = np.sum(ns)
    ts = np.empty_like(ns, dtype=float)
    print(ns)
    for i, n in enumerate(ns):
        cmd = f"calc_data({n})"
        t = timeit.timeit(
            cmd,
            setup="from __main__ import calc_data",
            number=10,
        )
        print(f"{n:<10}:{t:.5f}")
        v = t / n
        n_t = n_t - n
        print(f"\tit/s: {1/v:.3f}")
        eta = 10 * n_t * v
        print(f"\tETA: {eta:.2f}s")
        ts[i] = t

    np.savez("C:/Users/TheresaKunz/Python/AG_Güttel_GIT/sr-03-23/HSA_001_XXX/timing/time_EQC.npz", ns=ns, ts=ts)


if __name__ == "__main__":
    main()
