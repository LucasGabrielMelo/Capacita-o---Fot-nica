import numpy as np
import scipy.sparse as sp

# N -> Dimension of the grid
# h -> Taken step

#=========================== One dimensional derivates: ================================================

def D_center(N,h): #First order by central difference
    D = sp.eye(N, k = 1) - sp.eye(N, k = -1)
    D = D / (2*h)
    return D

def D_forward(N,h): #First order by forward difference
    D = sp.eye(N, k = 1) - sp.eye(N)
    D = D / h
    return D

def D_backward(N,h): #First order by backward difference
    D = sp.eye(N) - sp.eye(N, k = -1)
    D = D / h
    return D

def D2(N,h): #Second order
    D = sp.eye(N, k = -1) - 2*sp.eye(N) + sp.eye(N, k=1)
    D = D/ (h**2)
    return D

#=============== Two dimensional First Order matrices by forward difference: ===================================

def Dx_forward(N, h): 
    D = D_forward(N,h)
    I = sp.eye(N)
    D = sp.kron(I,D)
    return D

def Dy_forward(N, h):
    D = D_forward(N,h)
    I = sp.eye(N)
    D = sp.kron(D, I)
    return D

#=============== Two dimensional First Order matrices by backward difference: =================================

def Dx_backward(N, h):
    D = Dx_forward(N, h)
    D = ((-D).T).conjugate()
    return D

def Dy_left(N, h):
    D = Dy_forward(N, h)
    D = ((-D).T).conjugate()
    return D

#=============== Two dimensional First Order matrices by central difference: ================================

def Dx_center(N, h):
    D = D_center(N,h)
    I = sp.eye(N)
    D = sp.kron(I,D)
    return D

def Dy_center(N, h):
    D = D_center(N,h)
    I = sp.eye(N)
    D = sp.kron(D,I)
    return D

#======================== Two dimensional Second Order matrices: ===========================================

def Dx2(N, h):
    D = D2(N,h)
    I = sp.eye(N)
    D = sp.kron(I,D)
    return D

def Dy2(N, h):
    D = D2(N,h)
    I = sp.eye(N)
    D = sp.kron(D,I)
    return D

#=================================== Auxiliary functions: ==================================================

# Function that adjust the matrix for the boundary conditions

#boundary_conditions -> list with the indices of the lines in the matrix related to the boundary conditions
def adjust_matrix(L, boundary_conditions):
    L_new = L.tolil()
    for i in boundary_conditions:
        L_new[i,:] = 0
        L_new[i,i] = 1
    L_new = L_new.tocsr()
    return L_new