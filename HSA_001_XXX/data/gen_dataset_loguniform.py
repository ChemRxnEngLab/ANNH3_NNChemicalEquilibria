# Erzeugung von Gleichgewichtsdaten Ammoniaksynthese

# Importe / Bibliotheken
from pathlib import Path
import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
from scipy.constants import R
from scipy.stats import loguniform


# Stoechiometrische Koeffizienten Ammoniaksynthese
v_H2 = -3
v_N2 = -1
v_NH3 = 2

v = [v_H2, v_N2, v_NH3]

# Index der Stoffe in Arrays: [H2, N2, NH3]
index = np.array([0, 1, 2])
H2 = index[0]
N2 = index[1]
NH3 = index[2]

# Konstanten
p_0 = 1  # bar Standarddruck

# Parameter
num = 1_000_000  # Anzahl der Werte im Vektor

T = np.random.uniform(408.15, 1273.15, num)  # K Temperatur
p = loguniform.rvs(1, 500, num)  # bar Druck

# Stofffmengen zu Reaktionsbeginn
n_ges_0 = 1  # mol Gesamtstoffmenge zum Reaktionsbeginn
x_0 = np.random.dirichlet((1, 1, 1), num)  # 1 Stoffmengenanteile zu Reaktionsbeginn
n_H2_0 = x_0[:, 0] * n_ges_0  # mol Stoffmenge H2 Start
n_N2_0 = x_0[:, 1] * n_ges_0  # mol Stoffmenge N2 Start
n_NH3_0 = x_0[:, 2] * n_ges_0  # mol Stoffmenge NH3 Start

# Standardbildungsenthalpie delta_f_H_0_ref; NIST; [H2, N2, NH3]
delta_f_H_0_ref = np.array([0.0, 0.0, -45.90])  # kJ mol^-1 Standardtemperatur

data_path = Path.cwd() / "HSA_001_XXX" / "data"
data_file = data_path / "eq_dataset_loguniform.npz"

if not data_path.exists():
    print(f"{data_path} does not exist. Create it!")

if data_file.exists():
    print("orerriding existing data file")


# Shomate-Gleichungen
def shomate_S(T, stoff, A, B, C, D, E, F, G, H):
    t = T / 1000
    S = (
        A[stoff] * np.log(t)
        + B[stoff] * t
        + C[stoff] * t**2 / 2
        + D[stoff] * t**3 / 3
        - E[stoff] / (2 * t**2)
        + G[stoff]
    )
    return S


def shomate_H(T, stoff, A, B, C, D, E, F, G, H):
    t = T / 1000
    H_0 = (
        A[stoff] * t
        + B[stoff] * t**2 / 2
        + C[stoff] * t**3 / 3
        + D[stoff] * t**4 / 4
        - E[stoff] / t
        + F[stoff]
        - H[stoff]
        + delta_f_H_0_ref[stoff]
    )
    return H_0


# Standardbildungsenthalpie delta_f_H_0
def delta_f_H_0(T, stoff, A, B, C, D, E, F, G, H):
    # N2 oder H2 --> Standardbildungsenthalpie == 0
    if stoff != 2:
        delta_f_H_0 = 0.0
    # NH3 Berechnung der Standardbildungsenthalpie aus Shomate-Enthalpie
    else:
        delta_f_H_0 = (
            (v_NH3 / v_NH3) * shomate_H(T, NH3, A, B, C, D, E, F, G, H)
            + (v_N2 / v_NH3) * shomate_H(T, N2, A, B, C, D, E, F, G, H)
            + (v_H2 / v_NH3) * shomate_H(T, H2, A, B, C, D, E, F, G, H)
        )
    return delta_f_H_0


# Gleichgewichtsberechnung
def GGW(T, p, n_H2_0, n_N2_0, n_NH3_0):
    # Shomate Koeffizienten; NIST; [H2, N2, NH3]
    if T < 298:
        raise Warning("No data for this temperature available.")
    elif T < 500:
        A = np.array([33.066178, 28.98641, 19.99563])
        B = np.array([-11.363417, 1.853978, 49.77119])
        C = np.array([11.432816, -9.647459, -15.37599])
        D = np.array([-2.772874, 16.63537, 1.921168])
        E = np.array([-0.158558, 0.000117, 0.189174])
        F = np.array([-9.980797, -8.671914, -53.30667])
        G = np.array([172.707974, 226.4168, 203.8591])
        H = np.array([0.0, 0.0, -45.89806])
    elif T < 1000:
        A = np.array([33.066178, 19.50583, 19.99563])
        B = np.array([-11.363417, 19.88705, 49.77119])
        C = np.array([11.432816, -8.598535, -15.37599])
        D = np.array([-2.772874, 1.369784, 1.921168])
        E = np.array([-0.158558, 0.527601, 0.189174])
        F = np.array([-9.980797, -4.935202, -53.30667])
        G = np.array([172.707974, 212.3900, 203.8591])
        H = np.array([0.0, 0.0, -45.89806])
    elif T < 1400:
        A = np.array([18.563083, 19.50583, 19.99563])
        B = np.array([12.257357, 19.88705, 49.77119])
        C = np.array([-2.859786, -8.598535, -15.37599])
        D = np.array([0.268238, 1.369784, 1.921168])
        E = np.array([1.977990, 0.527601, 0.189174])
        F = np.array([-1.147438, -4.935202, -53.30667])
        G = np.array([156.288133, 212.3900, 203.8591])
        H = np.array([0.0, 0.0, -45.89806])
    else:
        raise Warning("Temperature is too high.")

    # Standardreaktionsenthalpie delta_R_H_0
    delta_R_H_0 = (
        v_H2 * delta_f_H_0(T, H2, A, B, C, D, E, F, G, H)
        + v_N2 * delta_f_H_0(T, N2, A, B, C, D, E, F, G, H)
        + v_NH3 * delta_f_H_0(T, NH3, A, B, C, D, E, F, G, H)
    ) * 1000  # J mol^-1

    # Standardreaktionsentropie delta_R_S_0
    delta_R_S_0 = (
        v_H2 * shomate_S(T, H2, A, B, C, D, E, F, G, H)
        + v_N2 * shomate_S(T, N2, A, B, C, D, E, F, G, H)
        + v_NH3 * shomate_S(T, NH3, A, B, C, D, E, F, G, H)
    )  # J mol^-1 K^-1

    # freie Standard Reaktionsenthalpie delta_R_G_0
    delta_R_G_0 = delta_R_H_0 - T * delta_R_S_0  # J mol^-1

    # allgemeine GGW-Konstante K_0
    K_0 = np.exp((-delta_R_G_0) / (T * R))  # 1

    # spezifische GGW-Konstante K_x
    K_x = K_0 * (p_0 / p) ** (
        sum(v)
    )  # 1 (Summe der stoechiometrischen Koeffizienten im Exponenten)

    # Numerische Loesung
    # Definition der Funktion
    def fun(xi):
        return (n_NH3_0 + 2 * xi) ** 2 * (n_ges_0 - 2 * xi) ** 2 - K_x * (
            n_H2_0 - 3 * xi
        ) ** 3 * (n_N2_0 - xi)

    # Bestimmung Startwert
    xi_0 = (-0.5 * n_NH3_0 + min(n_N2_0, 1 / 3 * n_H2_0)) / 2
    # xi_0 = np.full_like(K_x, xi_0) # bei Uebergabe T_array

    # Lösung Polynom
    sol = root(fun, xi_0)
    xi = sol.x  # mol Reaktionslaufzahl

    # Kontrolle: physikalisch moegliche Loesung?
    if xi < (-0.5 * n_NH3_0) or xi > min(n_N2_0, 1 / 3 * n_H2_0):
        # Fehlermeldung
        raise Warning("Impossible value for xi.")

    ## für Uebergabe von T_array
    ##Kontrolle: physikalisch moegliche Loesung?
    # for i in range(0, len(xi)):
    #     if xi[i] < (-0.5 * n_NH3_0) or xi[i] > min(n_N2_0, 1/3 * n_H2_0):
    #         #Fehlermeldung
    #         raise Warning("Impossible value for xi.")

    return (xi, K_x, K_0)


# Aufruf der GGW-Funktion und Berechnung der Stoffmengen im GGW
xi = np.zeros_like(n_H2_0)
for i in range(0, len(n_H2_0)):
    # print(i)
    if i % 1000 == 0:
        print(f"{i} of {num}")
    xi[i], _, _ = GGW(T[i], p[i], n_H2_0[i], n_N2_0[i], n_NH3_0[i])

# Berechnung der Stoffmengen im Gleichgewicht
n_H2 = xi * v_H2 + n_H2_0  # mol Stoffmenge H2 Gleichgewicht
n_N2 = xi * v_N2 + n_N2_0  # mol Stoffmenge N2 Gleichgewicht
n_NH3 = xi * v_NH3 + n_NH3_0  # mol Stoffmenge NH3 Gleichgewicht
n_ges = n_H2 + n_N2 + n_NH3  # mol Gesamtstoffmenge Gleichgewicht
x = (np.array([n_H2, n_N2, n_NH3]) / n_ges).T  # 1 Stoffmengenanteile im Gleichgewicht

# %% Speichern der GGW Daten
np.savez(data_file, T=T, p=p, x_0=x_0, x=x)
