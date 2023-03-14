#Erzeugung von Gleichgewichtsdaten Ammoniaksynthese

#Importe / Bibliotheken
import numpy as np

#Stöchiometrische Koeffizienten Ammoniaksynthese
v_H2 = -3
v_N2 = -1
v_NH3 = 2

v = ([v_H2, v_N2, v_NH3])

#Index der Stoffe in Arrays: [H2, N2, NH3]
index = np.array([0,1,2])
H2 = index[0]
N2 = index[1]
NH3 = index[2]

#
T = 300 # K Temperatur
p = 2 # bar Druck
R = 8.31448 # J mol^-1 K^-1 Ideale Gaskonstane
p_0 = 1 #bar Standarddruck

#Shomate Koeffizienten; NIST; [H2, N2, NH3]
#Achtung: Nur für Temperaturspanne zwischen 298 K und 500 K
#Noch ändern!
A = np.array([33.066178,28.98641,19.99563])
B = np.array([-11.363417,1.853978,49.77119])
C = np.array([11.432816,-9.647459,-15.37599])
D = np.array([-2.772874,16.63537	,1.921168])
E = np.array([-0.158558,0.000117,0.189174])
F = np.array([-9.980797,-8.671914,-53.30667])
G = np.array([172.707974,226.4168,203.8591])
H = np.array([0.0,0.0,-45.89806])


#Shomate-Gleichungen
def shomate_S (T, stoff):
    t = T / 1000
    S = A[stoff] * np.log(t) + B[stoff] * t + C[stoff] * t**2 / 2 + D[stoff] * t**3 / 3 - E[stoff] /(2 * t**2) + G[stoff]
    return S

#Standardreaktionsenthalpie delta_R_H_0
#delta_R_H_0 =

#Standardreaktionsentropie delta_R_S_0
delta_R_S_0 = v_H2 * shomate_S(T, H2) + v_N2 * shomate_S(T, N2) + v_NH3 * shomate_S(T, NH3)

#freie Standard Reaktionsenthalpie delta_R_G_0
#delta_R_G_0 = delta_R_H_0 - T * delta_R_S_0 # J mol^-1

#allgemeine GGW-Konstante K_0
#K_0 = np.exp(-delta_R_G_0 / (R*T)) # 1

#spezifische GGW-Konstante K_x
#K_x = K_0 * (p_0 / p)**(sum(v)) # 1 (Summe der stöchiometrischen Koeffizienten im Exponenten)
