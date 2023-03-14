#Erzeugung von Gleichgewichtsdaten Ammoniaksynthese

#Importe / Bibliotheken
import numpy as np

#
T = 250 # K Temperatur
R = 8,31448 # J mol^-1 K^-1 Ideale Gaskonstane

#Shomate-Gleichungen


delta_R_H_0 =
delta_R_S_0 =

#freie Standard Reaktionsenthalpie delta_R_G_0
delta_R_G_0 = delta_R_H_0 - T * delta_R_S_0 # J mol^-1

#allgemeine GGW-Konstante K_0
K_0 = np.exp(-delta_R_G_0 / (R*T)) # 1


