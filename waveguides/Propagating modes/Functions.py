# This python file have all functions used in this folder, to make the codes more readable

import numpy as np 

# ====== Functions to plot the normalized frequency and the normalized propagation constant =================

def delta(nf, ns, nc): # Return the asymmetry parameter
    numerator = ns**2 - nc**2
    denominator = nf**2 - nc**2
    d = numerator / denominator
    return d

def Vcm(V, nf = 3.5, ns = 1.5, nc = 1): # Return the number of propagating modes in the waveguide
    d = delta(nf, ns, nc)
    arctan = np.arctan(np.sqrt(d))
    V_max = np.max(V)
    M = 0
    Vc = 0
    while True:
        Vc = (M*np.pi + arctan) / 2
        if Vc <= V_max:
            M += 1
        else:
            break
    return M # Return the maximum index of the propagating modes

def Vcm_TM_modes(V, nf = 3.5, ns = 1.5, nc = 1):
    d = delta(nf, ns, nc)
    pc = (nf/nc)**2
    arctan = np.arctan(pc * np.sqrt(d))
    V_max = np.max(V)
    M = 0
    Vc = 0
    while True:
        Vc = (M*np.pi + arctan) / 2
        if Vc <= V_max:
            M += 1
        else:
            break
    return M

def V_TE(b, M, nf = 3.5, ns = 1.5, nc = 1):
    v = []
    d = delta(nf, ns, nc)
    for m in range(0, M, 1):
        arctan1 = np.arctan(np.sqrt((b+d)/(1-b)))
        arctan2 = np.arctan(np.sqrt(b/(1-b)))
        vm = (m*np.pi + arctan1 + arctan2) / (2 * np.sqrt(1-b))
        v.append(vm)
    return np.array(v) 

def V_TM(b, M, nf = 3.5, ns = 1.5, nc = 1):
    v = []
    d = delta(nf, ns, nc)
    pc = (nf/nc)**2
    ps = (nf/ns)**2
    for m in range(0, M, 1):
        arctan1 = np.arctan(pc * np.sqrt((b+d)/(1-b)))
        arctan2 = np.arctan(ps * np.sqrt(b/(1-b)))
        vm = (m*np.pi + arctan1 + arctan2) / (2 * np.sqrt(1-b))
        v.append(vm)
    return np.array(v)

# ======= Functions to localize the effective indices for a given normalized frequency ==============

def b_TE_modes(v, nf = 3.5, ns = 1.5, nc = 1): # Localize the normalized propagation constants of TE modes for a given normalized frequency
    M = Vcm(v, nf, ns, nc) 
    b = np.linspace(0, 0.99, 100)
    V_te = V_TE(b, M, nf, ns, nc)
    b_TE = []
    for i in range(0, M, 1):
        b_TE.append(np.interp(v, V_te[i], b))
    return np.array(b_TE)

def b_TM_modes(v, nf = 3.5, ns = 1.5, nc = 1):
    M = Vcm_TM_modes(v, nf, ns, nc)
    b = np.linspace(0, 0.99, 100)
    V_tm = V_TM(b, M, nf, ns, nc)
    b_TM = []
    for i in range(0, M, 1):
        b_TM.append(np.interp(v, V_tm[i], b))
    return np.array(b_TM)
