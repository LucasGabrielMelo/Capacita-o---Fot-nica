import numpy as np
import scipy.sparse as sp

# N -> Dimensão da malha
# h -> Passo dado

#================ Matrizes de primeira ordem via aproximação progressiva: ===========================

def Dx_right(N, h): 
    N2 = N**2
    D = -sp.eye(N2) + sp.eye(N2, k=1)
    D = D.tolil()
    for i in range(N-1, N2-N, N):
        D[i, i+1] = 0
    D = D.tocsr()
    D = D / h
    return D

def Dy_right(N, h):
    N2 = N**2
    D = -sp.eye(N2) + sp.eye(N2, k=N)
    D = D / h
    return D

#================ Matrizes de primeira ordem via aproximação regressiva: ===========================

def Dx_left(N, h):
    D = Dx_right(N, h)
    D = ((-D).T).conjugate()
    return D

def Dy_left(N, h):
    D = Dy_right(N, h)
    D = ((-D).T).conjugate()
    return D

#================ Matrizes de segunda ordem via aproximação central: ================================

def Dx_center(N, h):
    N2 = N**2
    D = -sp.eye(N2, k = -1) + sp.eye(N2, k = 1)
    D = D.tolil()
    for i in range(N-1,N2-N, N):
        D[i+1,i] = 0
        D[i,i+1] = 0
    D = D.tocsr()
    D = D / (2*h)
    return D

def Dy_center(N, h):
    N2 = N**2
    D = sp.eye(N2, k = N) - sp.eye(N2, k = -N)
    D = D / (2*h)
    return D

#============================== Matrizes de segunda ordem: ===========================================

def Dx2(N, h):
    N2 = N**2
    D = sp.eye(N2,k=-1) - 2*sp.eye(N2) + sp.eye(N2,k=1)
    D = D.tolil()
    for i in range(N-1,N2-N, N):
        D[i+1,i] = 0
        D[i,i+1] = 0
    D = D.tocsr()
    D = D / (h**2)
    return D

def Dy2(N, h):
    N2 = N**2
    D = sp.eye(N2, k = - N) - 2*sp.eye(N2) + sp.eye(N2, k = N)
    D = D / (h**2)
    return D