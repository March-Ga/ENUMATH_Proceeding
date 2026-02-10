import numpy as np
from scipy.sparse.linalg import factorized

# from utils import solid_assembly, fluid_assembly_CF,\
#       fluid_solver_CF, assembly_T_IB, assembly_T_ID

from utils import *


def solve_ID(nx_f, ny_f, dx_f, dy_f, Nt, vel0_f, lx_s, ly_s, nx_s, ny_s, dx_s, dy_s, d0_x_s, d0_y_s, mu_s, lam_s, dt, Nb, nu_f, rho_f, G_f, rho_s, tol=1e-6, minit=0, maxit=100): 
    beta = 0.25
    gamma = 0.5
    
    p_f = np.zeros([ny_f + 2, nx_f + 2]) 
    u_f = np.zeros_like(p_f)
    v_f = np.zeros_like(p_f)

    usol_f = []
    vsol_f = []
    psol_f = []

    usol_f.append(u_f.copy())
    vsol_f.append(v_f.copy())
    psol_f.append(p_f.copy())

    usol_s = []
    vsol_s = []
    asol_s = []

    u_s = np.zeros(Nb * 2)
    v_s = np.zeros(Nb * 2)
    a_s = np.zeros(Nb * 2)

    usol_s.append(u_s.copy())
    vsol_s.append(v_s.copy())
    asol_s.append(a_s.copy())

    lam = np.zeros(2 * Nb)
    lam_old = np.zeros_like(lam)      

    M_el, A_s, b_s, P, T, M, bdry = solid_assembly(lx_s, ly_s, nx_s, ny_s, dx_s, dy_s, lam_s, mu_s, np.array([-2, -2, -2, -2]))
    Af, bf = fluid_assembly_CF(nx_f, ny_f, dx_f, dy_f, G_f)

    tot_it = 0
    for t in range(Nt):

        Tx, Ty = assembly_T_ID(u_s[:Nb] + P[0,:] + d0_x_s, u_s[Nb:] + P[1,:] + d0_y_s, dx_f, dy_f, nx_s, ny_s, nx_f, ny_f, T) 
          
        k = 0
        err = 1e5
        un = usol_s[t] + dt * vsol_s[t] + dt * dt * (0.5 - beta) * asol_s[t] 
        vn = vsol_s[t] + dt * (1.0 - gamma) * asol_s[t]
        while err > tol and k < maxit or k < minit:

            v_s_x = Tx @ u_f[1:-1, 1:].ravel() 
            v_s_y = Ty @ v_f[1:, 1:-1].ravel() 
            v_s = np.concatenate((v_s_x, v_s_y))

            a_s = (v_s - vn) /(gamma * dt)
            u_s = un + beta * dt * dt * a_s

            lam = (rho_s - rho_f) * M_el @ a_s + A_s @ u_s                                
 
            F_x = - Tx.T @ lam[:Nb] 
            F_y = - Ty.T @ lam[Nb:] 

            F_x = F_x.reshape((ny_f, nx_f + 1))       
            F_y = F_y.reshape((ny_f + 1, nx_f)) 

            u_f, v_f, p_f = fluid_solver_CF(usol_f[t].copy(), vsol_f[t].copy(), nx_f, ny_f, dx_f, dy_f, dt, nu_f, F_x, F_y, Af, bf, vel0_f, rho_f)

            err = np.linalg.norm(lam - lam_old, ord = 2)
            lam_old = lam.copy()
            k += 1
        print(f'time step: {t}/{Nt}, iterations: {k}, error: {err}')
        tot_it += k
        usol_f.append(u_f.copy())
        vsol_f.append(v_f.copy())
        psol_f.append(p_f.copy())

        usol_s.append(u_s.copy())
        vsol_s.append(v_s.copy())
        asol_s.append(a_s.copy())

    return usol_f, vsol_f, psol_f, usol_s, vsol_s, asol_s, P, T, bdry, tot_it


def solve_IB(nx_f, ny_f, dx_f, dy_f, Nt, vel0_f, lx_s, ly_s, nx_s, ny_s, dx_s, dy_s, d0_x_s, d0_y_s, mu_s, lam_s, dt, Nb, nu_f, rho_f, G_f, rho_s, tol=1e-6, minit=0, maxit=100): 
    beta = 0.25
    gamma = 0.5

    p_f = np.zeros([ny_f + 2, nx_f + 2]) 
    u_f = np.zeros_like(p_f)
    v_f = np.zeros_like(p_f)

    usol_f = []
    vsol_f = []
    psol_f = []

    usol_f.append(u_f.copy())
    vsol_f.append(v_f.copy())
    psol_f.append(p_f.copy())

    usol_s = []
    vsol_s = []
    asol_s = []

    u_s = np.zeros(Nb * 2)
    v_s = np.zeros(Nb * 2)
    a_s = np.zeros(Nb * 2)

    usol_s.append(u_s.copy())
    vsol_s.append(v_s.copy())
    asol_s.append(a_s.copy())

    lam = np.zeros(2 * Nb)
    lam_old = np.zeros_like(lam)      

    M_el, A_s, b_s, P, T, M, bdry = solid_assembly(lx_s, ly_s, nx_s, ny_s, dx_s, dy_s, lam_s, mu_s, np.array([-2, -2, -2, -2]))
    R = (rho_s - rho_f) * M_el/(gamma * dt) + (beta * dt / gamma) * A_s
    dirichlet = np.unique(bdry[2:4, :])

    for k in range(len(dirichlet)):
        i = dirichlet[k]
        R[i, :] = 0.0
        R[i, i] = 1.0
    
        R[i + Nb, :] = 0.0
        R[i + Nb, i + Nb] = 1.0

    R = R.tocsc()
    solve_s = factorized(R)

    Af, bf = fluid_assembly_CF(nx_f, ny_f, dx_f, dy_f, G_f)

    tot_it = 0
    for t in range(Nt):
        Tx, Ty = assembly_T_IB(u_s[:Nb] + P[0,:] + d0_x_s, u_s[Nb:] + P[1,:] + d0_y_s, dx_f, dy_f, nx_s, ny_s, nx_f, ny_f, bdry) 
 
        k = 0
        err = 1e5
        un = usol_s[t] + dt * vsol_s[t] + dt * dt * (0.5 - beta) * asol_s[t] 
        vn = vsol_s[t] + dt * (1.0 - gamma) * asol_s[t]
        rhs = (rho_s - rho_f) * (1/(gamma * dt)) * M_el * vn - A_s * (un - (beta * dt / gamma)* vn)
        while err > tol and k < maxit or k < minit:
            
            v_s_x = Tx @ u_f[1:-1, 1:].ravel() 
            v_s_y = Ty @ v_f[1:, 1:-1].ravel() 

            for l in range(len(dirichlet)):
                i = dirichlet[l]
            
                rhs[i] = v_s_x[i]
                rhs[i + Nb] = v_s_y[i]
            
            v_s = solve_s(rhs) 

            a_s = (v_s - vn) /(gamma * dt)
            u_s = un + beta * dt * dt * a_s
        
            lam = ((rho_s - rho_f) * M_el @ a_s + A_s @ u_s)                                    
    
            F_x = - Tx.T @ lam[:Nb] 
            F_y = - Ty.T @ lam[Nb:] 


            F_x = F_x.reshape((ny_f, nx_f + 1))     
            F_y = F_y.reshape((ny_f + 1, nx_f)) 

            u_f, v_f, p_f = fluid_solver_CF(usol_f[t].copy(), vsol_f[t].copy(), nx_f, ny_f, dx_f, dy_f, dt, nu_f, F_x, F_y, Af, bf, vel0_f, rho_f)
 
            err = np.linalg.norm(lam - lam_old, ord = 2)
            lam_old = lam.copy()
        
            k += 1
        print(f'time step: {t}/{Nt}, iterations: {k}, error: {err}')
        tot_it += k
        usol_f.append(u_f.copy())
        vsol_f.append(v_f.copy())
        psol_f.append(p_f.copy())

        usol_s.append(u_s.copy())
        vsol_s.append(v_s.copy())
        asol_s.append(a_s.copy())

    return usol_f, vsol_f, psol_f, usol_s, vsol_s, asol_s, P, T, bdry, tot_it