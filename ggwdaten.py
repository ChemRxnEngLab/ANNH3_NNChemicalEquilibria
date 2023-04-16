#Erzeugung von Gleichgewichtsdaten Ammoniaksynthese

#Importe / Bibliotheken
import numpy as np

#Stoechiometrische Koeffizienten Ammoniaksynthese
v_H2 = -3
v_N2 = -1
v_NH3 = 2

v = ([v_H2, v_N2, v_NH3])

#Index der Stoffe in Arrays: [H2, N2, NH3]
index = np.array([0,1,2])
H2 = index[0]
N2 = index[1]
NH3 = index[2]

#Konstanten
R = 8.31448 # J mol^-1 K^-1 Ideale Gaskonstane
p_0 = 1 # bar Standarddruck

# Parameter
#T = 300 # K Temperatur
T_array = np.array([300, 350]) # K Temperatur
p = 2 # bar Druck

n_H2 = 5 # mol Stoffmenge H2
n_N2 = 1/3 * n_H2 # mol Stoffmenge N2 (stöchiometrisch)


#Shomate Koeffizienten; NIST; [H2, N2, NH3]
#Achtung: Nur für Temperaturspanne zwischen 298 K und 500 K
#Noch aendern! (if else)
A = np.array([33.066178,28.98641,19.99563])
B = np.array([-11.363417,1.853978,49.77119])
C = np.array([11.432816,-9.647459,-15.37599])
D = np.array([-2.772874,16.63537	,1.921168])
E = np.array([-0.158558,0.000117,0.189174])
F = np.array([-9.980797,-8.671914,-53.30667])
G = np.array([172.707974,226.4168,203.8591])
H = np.array([0.0,0.0,-45.89806])

#Standardbildungsenthalpie delta_f_H_0_ref; NIST; [H2, N2, NH3]
delta_f_H_0_ref = np.array([0.0, 0.0, -45.90]) # kJ mol^-1

#Shomate-Gleichungen
def shomate_S (T, stoff):
    t = T / 1000
    S = A[stoff] * np.log(t) + B[stoff] * t + C[stoff] * t**2 / 2 + D[stoff] * t**3 / 3 - E[stoff] /(2 * t**2) + G[stoff]
    return S

def shomate_H (T, stoff):
    t = T / 1000
    H_0 = A[stoff] * t + B[stoff] * t**2 / 2 + C[stoff] * t**3 / 3 + D[stoff] * t**4 / 4 - E[stoff] / t + F[stoff] - H[stoff] + delta_f_H_0_ref[stoff] 
    return H_0

#Standardbildungsenthalpie delta_f_H_0
def delta_f_H_0(T, stoff):
    #N2 oder H2 --> Standardbildungsenthalpie == 0
    if stoff != 2:
        delta_f_H_0 = 0.0
    #NH3 Berechnung der Standardbildungsenthalpie aus Shomate-Enthalpie
    else:
        delta_f_H_0 = v_NH3 * shomate_H(T, NH3) + v_N2 * shomate_H(T,N2) + v_H2 * shomate_H(T,H2)
    return delta_f_H_0

#Standardreaktionsenthalpie delta_R_H_0
delta_R_H_0 = np.zeros(len(T_array))
for i in range (0, len(T_array)):
    T = T_array[i]
    delta_R_H_0[i] = (v_H2 * delta_f_H_0(T,H2) + v_N2 * delta_f_H_0(T, N2) + v_NH3 * delta_f_H_0(T,NH3)) * 1000 # J mol^-1

#Standardreaktionsentropie delta_R_S_0
delta_R_S_0 = np.zeros(len(T_array))
for i in range (0, len(T_array)):
    T = T_array[i]
    delta_R_S_0[i] = v_H2 * shomate_S(T, H2) + v_N2 * shomate_S(T, N2) + v_NH3 * shomate_S(T, NH3) # J mol^-1 K^-1

#freie Standard Reaktionsenthalpie delta_R_G_0
delta_R_G_0 = delta_R_H_0 - T_array * delta_R_S_0 # J mol^-1

#allgemeine GGW-Konstante K_0
K_0 = np.exp((-delta_R_G_0) / (T_array * R)) # 1

#spezifische GGW-Konstante K_x
K_x = K_0 * (p_0 / p)**(sum(v)) # 1 (Summe der stoechiometrischen Koeffizienten im Exponenten)

# =============================================================================
# #Berechnung von Stoffmenge Ammoniak bei gegebenen Stoffmengen von H2 und N2
# #Analytische Loesung (Gleichung 3. Grades)
# #Koeffizienten
# a = 
# b =
# c =
# 
# # Substitution
# z = x
# =============================================================================

#
# =============================================================================
# #Analytische Loesung (Gleichung 4. Grades)
# #Koeffizienten
# a = 1
# b = 2 * (n_H2 * n_N2)
# c = (n_H2 + n_N2)**2
# d = 0
# e = -K_x * n_H2**3 * n_N2
# 
# p = (8 * a * c - 3 * b**2) / (8 * a**3)
# q = (b**3 - 4 * a * b * c + 8 * a**2 * d) / (8 * a**3)
# 
# delta_0 = c**2 - 3 * b * d + 12 * a * e
# delta_1 = 2 * c**3 - 9 * b * c * d + 27 * b**2 * e + 27 * a * d**2 - 72 * a * c * e
# 
# Q = ((delta_1 + (delta_1**2 - 4 * delta_0**3)**0.5) / 2)**(1/3)
# S = 0.5 * (-2 / 3 * p + 1 / (3 * a) * (Q + (delta_0 / Q)))**0.5
# 
# #moegliche Loesungen für Stoffmenge von Ammoniak
# n_NH3 = np.zeros((4,len(T_array)))
# n_NH3[0] = -b / (4 * a) - S + 0.5 *(-4 * S**2 - 2 * p + q / S)**0.5
# n_NH3[1] = -b / (4 * a) - S - 0.5 *(-4 * S**2 - 2 * p + q / S)**0.5
# n_NH3[2] = -b / (4 * a) + S + 0.5 *(-4 * S**2 - 2 * p + q / S)**0.5
# n_NH3[3] = -b / (4 * a) + S - 0.5 *(-4 * S**2 - 2 * p + q / S)**0.5
# print(n_NH3)
# 
# 
# #Loeschen der physikalisch nicht moeglichen Loesungen
# k = 3
# for j in range (0, len(T_array)):
#     for i in range(k,0-1,-1):
#         if n_NH3[i,j] < 0:
#             n_NH3 = np.delete(n_NH3,[i,j])
#         elif n_NH3[i,j] > (-v_NH3 / v_H2 * n_H2):
#             n_NH3 = np.delete(n_NH3,[i,j])
#         elif n_NH3[i,j] > (-v_NH3 / v_N2 * n_N2):
#             n_NH3 = np.delete(n_NH3,[i,j])
#     print(n_NH3)
# =============================================================================

        
        







