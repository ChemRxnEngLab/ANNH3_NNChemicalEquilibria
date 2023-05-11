#Erzeugung von Gleichgewichtsdaten Ammoniaksynthese

#Importe / Bibliotheken
import numpy as np
from scipy.optimize import root


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
T = np.array([300, 350]) # K Temperatur
#p = 2 # bar Druck
p = np.array([1.5, 2]) # bar Druck

T = np.random.uniform(600,800 + 1,num) # K Temperatur
p = np.random.uniform(100,250 + 1,num) # bar Druck

#Stofffmengen zu Reaktionsbeginn
n_ges_0 = 1 # mol Gesamtstoffmenge zum Reaktionsbeginn
x_0 = np.random.dirichlet((1,1,1),num) # 1 Stoffmengenanteile zu Reaktionsbeginn
n_H2_0 = x_0[:,0] * n_ges_0 # mol Stoffmenge H2 Start
n_N2_0 = x_0[:,1] * n_ges_0 # mol Stoffmenge N2 Start
n_NH3_0 = x_0[:,2] * n_ges_0 # mol Stoffmenge NH3 Start

#Shomate Koeffizienten; NIST; [H2, N2, NH3]
A = np.array([33.066178,28.98641,19.99563])
B = np.array([-11.363417,1.853978,49.77119])
C = np.array([11.432816,-9.647459,-15.37599])
D= np.array([-2.772874,16.63537	,1.921168])
E = np.array([-0.158558,0.000117,0.189174])
F = np.array([-9.980797,-8.671914,-53.30667])
G = np.array([172.707974,226.4168,203.8591])
H = np.array([0.0,0.0,-45.89806])
# A = np.zeros((num,3))
# B = C = D = E = F = G = H = np.zeros((num))
# for i in range(0,num-1):
#     if T[i] < 298:
#         raise Warning ("No data for this temperature available.")
#     elif T[i] < 500:
#         A = np.append(A,[[33.066178],[28.98641],[19.99563]],axis=0)
#         B[i] = np.array([-11.363417,1.853978,49.77119])
#         C[i] = np.array([11.432816,-9.647459,-15.37599])
#         D[i] = np.array([-2.772874,16.63537	,1.921168])
#         E[i] = np.array([-0.158558,0.000117,0.189174])
#         F[i] = np.array([-9.980797,-8.671914,-53.30667])
#         G[i] = np.array([172.707974,226.4168,203.8591])
#         H[i] = np.array([0.0,0.0,-45.89806])
#     elif T < 1000:
#         A[i] = np.array([33.066178,19.50583,19.99563])
#         B[i] = np.array([-11.363417,19.88705,49.77119])
#         C[i] = np.array([11.432816,-8.598535,-15.37599])
#         D[i] = np.array([-2.772874,1.369784	,1.921168])
#         E[i] = np.array([-0.158558,0.527601,0.189174])
#         F[i] = np.array([-9.980797,-4.935202,-53.30667])
#         G[i] = np.array([172.707974,212.3900,203.8591])
#         H[i] = np.array([0.0,0.0,-45.89806])
#     elif T < 1400:
#         A[i] = np.array([18.563083,19.50583,19.99563])
#         B[i] = np.array([12.257357,19.88705,49.77119])
#         C[i] = np.array([-2.859786,-8.598535,-15.37599])
#         D[i] = np.array([0.268238,1.369784	,1.921168])
#         E[i] = np.array([1.977990,0.527601,0.189174])
#         F[i] = np.array([-1.147438,-4.935202,-53.30667])
#         G[i] = np.array([156.288133,212.3900,203.8591])
#         H[i] = np.array([0.0,0.0,-45.89806])
#     else:
#         raise Warning ("Temperature is too high.")
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
        delta_f_H_0 = (v_NH3 / v_NH3) * shomate_H(T, NH3) + (v_N2 / v_NH3) * shomate_H(T,N2) + (v_H2 / v_NH3) * shomate_H(T,H2)
    return delta_f_H_0

#Gleichgewichtsberechnung
def GGW(T, p, n_H2_0, n_N2_0, n_NH3_0):
    
    #Standardreaktionsenthalpie delta_R_H_0
    delta_R_H_0 = (v_H2 * delta_f_H_0(T,H2) + v_N2 * delta_f_H_0(T, N2) + v_NH3 * delta_f_H_0(T,NH3)) * 1000 # J mol^-1
    
    #Standardreaktionsentropie delta_R_S_0
    delta_R_S_0 = v_H2 * shomate_S(T, H2) + v_N2 * shomate_S(T, N2) + v_NH3 * shomate_S(T, NH3) # J mol^-1 K^-1
    
    #freie Standard Reaktionsenthalpie delta_R_G_0
    delta_R_G_0 = delta_R_H_0 - T * delta_R_S_0 # J mol^-1
    
    #allgemeine GGW-Konstante K_0
    K_0 = np.exp((-delta_R_G_0) / (T * R)) # 1
    
    #spezifische GGW-Konstante K_x
    K_x = K_0 * (p_0 / p)**(sum(v)) # 1 (Summe der stoechiometrischen Koeffizienten im Exponenten)
    
    #Numerische Loesung
    #Definition der Funktion
    def fun(xi):
        return (n_NH3_0 + 2 * xi)**2 * (n_ges_0 - 2 * xi)**2 - K_x * (n_H2_0 - 3 * xi)**3 * (n_N2_0 - xi)

    #Bestimmung Startwert
    xi_0 = (-0.5 * n_NH3_0 + min(n_N2_0, 1/3 * n_H2_0)) / 2

    #Lösung Polynom
    sol = root(fun, xi_0)
    xi = sol.x # mol Reaktionslaufzahl
    
    #Kontrolle: physikalisch moegliche Loesung?
    if xi < (-0.5 * n_NH3_0) or xi > min(n_N2_0, 1/3 * n_H2_0):
        #Fehlermeldung
        raise Warning("Impossible value for xi.")
    
    return(xi)

#Aufruf der GGW-Funktion und Berechnung der Stoffmengen im GGW
xi = np.zeros(len(n_H2_0))
for i in range(0, len(n_H2_0)):
    xi[i] = GGW(T[i], p[i], n_H2_0[i], n_N2_0[i], n_NH3_0[i])
    #Berechnung der Stoffmengen im Gleichgewicht
    
n_H2 = xi * v_H2 + n_H2_0 # mol Stoffmenge H2 Gleichgewicht
n_N2 = xi * v_N2 + n_N2_0 # mol Stoffmenge N2 Gleichgewicht
n_NH3 = xi * v_NH3 + n_NH3_0 # mol Stoffmenge NH3 Gleichgewicht


# #Standardreaktionsentropie delta_R_S_0
# delta_R_S_0 = np.zeros(len(T_array))
# for i in range (0, len(T_array)):
#     T = T_array[i]
#     delta_R_S_0[i] = v_H2 * shomate_S(T, H2) + v_N2 * shomate_S(T, N2) + v_NH3 * shomate_S(T, NH3) # J mol^-1 K^-1

# #freie Standard Reaktionsenthalpie delta_R_G_0
# delta_R_G_0 = delta_R_H_0 - T_array * delta_R_S_0 # J mol^-1

# #allgemeine GGW-Konstante K_0
# K_0 = np.exp((-delta_R_G_0) / (T_array * R)) # 1

# #spezifische GGW-Konstante K_x
# # K_x = np.zeros((len(K_0), len(p_array)))
# # for i in range(0,len(K_0)):
# #     K_x[i] = K_0[i]* (p_0 / p_array)**(sum(v)) # 1 (Summe der stoechiometrischen Koeffizienten im Exponenten)

# K_x = K_0 * (p_0 / p)**(sum(v)) # 1 (Summe der stoechiometrischen Koeffizienten im Exponenten)


# #Numerische Loesung
# #Definition der Funktion
# n_ges_0 = n_H2_0 + n_N2_0 + n_NH3_0 # mol Gesamtstoffmenge
# def fun(xi):
#     return (n_NH3_0 + 2 * xi)**2 * (n_ges_0 - 2 * xi)**2 - K_x * (n_H2_0 - 3 * xi)**3 * (n_N2_0 - xi)

# #Bestimmung Startwert
# xi_0 = (-0.5 * n_NH3_0 + min(n_N2_0, 1/3 * n_H2_0)) / 2
# xi_0 = np.full_like(K_x, xi_0)
# #Lösung Polynom
# sol = root(fun, xi_0)
# xi = sol.x #mol Reaktionslaufzahl

# #Kontrolle: pyhiskalisch moegliche Loesung?
# for i in range(0, len(xi)):
#     j = 0 # Zähler while-Schleife resetten
#     while xi[i] < (-0.5 * n_NH3_0) or xi[i] > min(n_N2_0, 1/3 * n_H2_0):
#         #Berechnung xi mit anderem Startwert
#         xi_0[i] = (-0.5 * n_NH3_0 + min(n_N2_0, 1/3 * n_H2_0)) / (4 * (j+1))
#         sol = root(fun, xi_0)
#         xi = sol.x #mol Reaktionslaufzahl

# #Berechnung der Stoffmengen im Gleichgewicht
# n_H2 = xi * v_H2 + n_H2_0 # mol Stoffmenge H2 Gleichgewicht
# n_N2 = xi * v_N2 + n_N2_0 # mol Stoffmenge N2 Gleichgewicht
# n_NH3 = xi * v_NH3 + n_NH3_0 # mol Stoffmenge NH3 Gleichgewicht


    

# #Lösung durch Wurzelausdrücke
# # =============================================================================
# #Analytische Loesung (Gleichung 4. Grades)
# #Koeffizienten
# a = K_x * v_N2 * v_H2**3 - (v_NH3**2 * sum(v)**2)
# b = K_x * (n_N2_0 * v_H2**3 + 3 * v_N2 * n_H2_0 * v_H2**2) - (2 * n_NH3_0 * v_NH3 * sum(v)**2 + 2 * n_ges_0 * v_NH3**2 * sum(v))
# c = K_x * (3 * n_N2_0 * n_H2_0 * v_H2**2 + 3 * n_H2_0**2 * v_H2**2) - (n_NH3_0**2 * sum(v)**2 + 4 * n_NH3_0 * v_NH3 * n_ges_0 * sum(v) + v_NH3**2 * n_ges_0**2)
# d = K_x * (3 * n_N2_0 * n_H2_0**2 * v_H2 + v_N2 * n_H2_0**3) - (2 * n_NH3_0**2 * n_ges_0 * sum(v) + 2 * n_NH3_0 * v_NH3 * n_ges_0**2)
# e = K_x * (n_N2_0 * n_H2_0**3) - (n_NH3_0**2 * n_ges_0**2)


# # =============================================================================
# # a = 1
# # b = B / A
# # c = C / A
# # d = D / A
# # e = E / A
# # =============================================================================
# p = (8 * a * c - 3 * b**2) / (8 * a**3)
# q = (b**3 - 4 * a * b * c + 8 * a**2 * d) / (8 * a**3)

# delta_0 = c**2 - 3 * b * d + 12 * a * e
# delta_1 = 2 * c**3 - 9 * b * c * d + 27 * b**2 * e + 27 * a * d**2 - 72 * a * c * e


# Q = ((delta_1 + (delta_1**2 - 4 * delta_0**3)**0.5) / 2)**(1/3) # Achtung negative Wurzel
# S = 0.5 * (-2 / 3 * p + 1 / (3 * a) * (Q + (delta_0 / Q)))**0.5

# #moegliche Loesungen für Stoffmenge von Ammoniak
# xi = np.zeros((4,len(T_array)))
# xi[0] = -b / (4 * a) - S + 0.5 *(-4 * S**2 - 2 * p + q / S)**0.5
# xi[1] = -b / (4 * a) - S - 0.5 *(-4 * S**2 - 2 * p + q / S)**0.5
# xi[2] = -b / (4 * a) + S + 0.5 *(-4 * S**2 - 2 * p + q / S)**0.5
# xi[3] = -b / (4 * a) + S - 0.5 *(-4 * S**2 - 2 * p + q / S)**0.5
# print(xi)
# # =============================================================================
 
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

        
        







