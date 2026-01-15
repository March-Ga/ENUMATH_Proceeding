import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import splu
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse.linalg import spsolve

from FSI_DC_nv import solid_assembly
from FSI_DC_nv import fluid_assembly
from FSI_DC_nv import fluid_solver_nv
from FSI_DC_nv import assembly_B_ID

# SOLID INERTIA - ELASTODYNAMICS VERSION (Newmark-beta scheme)

def solve(lx_f, ly_f, nx_f, ny_f, dx_f, dy_f, Nt, vel0_f, lx_s, ly_s, nx_s, ny_s, dx_s, dy_s, d0_x_s, d0_y_s, mu_s, lam_s, dt, Nb, nu_f, rho_f, rho_s, tol=1e-6, minit=0, maxit=100): 

    p_f = np.zeros([ny_f + 2, nx_f + 2])      # N.B. i e j sono invertiti !
    u_f = np.zeros_like(p_f)
    v_f = np.zeros_like(p_f)

    usol_f = []
    vsol_f = []
    psol_f = []

    usol_f.append(u_f.copy())
    vsol_f.append(v_f.copy())
    psol_f.append(p_f.copy())

    Ut, Ub, Vl, Vr = vel0_f

    usol_s = []
    vsol_s = []
    asol_s = []

    u1_s = np.zeros(Nb)
    u2_s = np.zeros(Nb)
    u_s = np.concatenate((u1_s, u2_s))

    #u_s = np.zeros(Nb * 2)
    v_s = np.zeros(Nb * 2)
    a_s = np.zeros(Nb * 2)

    usol_s.append(u_s.copy())
    vsol_s.append(v_s.copy())
    asol_s.append(a_s.copy())

    lam = np.zeros(2 * Nb)
    lam_x = np.zeros(Nb)
    lam_y = np.zeros(Nb)

    R_s, M_el, A_s, b_s, P, T, M, bdry = solid_assembly(lx_s, ly_s, nx_s, ny_s, dx_s, dy_s, lam_s, mu_s, dt)
    Af, _ = fluid_assembly(nx_f, ny_f, dx_f, dy_f)
    
    tot_it = 0
    for t in range(Nt):
        u_f_small = u_f[1:-1, 1:]
        v_f_small = v_f[1:, 1:-1]

        # One coupling operator per timestep -> OK for small dt
        Bx, By, Bxx, Byy = assembly_B_ID(u1_s + P[0,:] + d0_x_s, u2_s + P[1,:] + d0_y_s, dx_f, dy_f, nx_s, ny_s, nx_f, ny_f, T) 
 
        lam_old = np.zeros_like(lam)

        lam_x = np.zeros(Nb)
        lam_y = np.zeros(Nb)

        #Picard iteration solver
        k = 0
        err = 1e5

        while err > tol and k < maxit or k < minit:

            # Get solid velocity from fluid interface (kinematic condition)
            v1_s = Bx @ u_f_small.ravel() 
            v2_s = By @ v_f_small.ravel() 
            v_s = np.concatenate((v1_s, v2_s))

            a_s = (v_s - vsol_s[t])/dt
            u_s = usol_s[t] + dt * v_s #+ dt * dt * 0.5 * a_s

            lam = (rho_s - rho_f) * M_el @ a_s + A_s @ u_s                                
    
            lam_x = lam[:Nb] 
            lam_y = lam[Nb:] 
            
            # Transfer force to fluid
            F_x = - Bx.T @ lam_x  
            F_y = - By.T @ lam_y  

            F_x = F_x.reshape((ny_f, nx_f + 1))       
            F_y = F_y.reshape((ny_f + 1, nx_f))

            u_f, v_f, p_f = fluid_solver_nv(usol_f[t].copy(), vsol_f[t].copy(), nx_f, ny_f, dx_f, dy_f, dt, nu_f, F_x, F_y, Af, vel0_f, rho_f)
        
            u_f_small = u_f[1:-1, 1:]
            v_f_small = v_f[1:, 1:-1]

            err = np.linalg.norm(lam - lam_old, ord = 2)
            # print(err)
            lam_old = lam.copy()
            # print(err)

            k += 1
        print(k,err)

        if t % 100 == 0:
            print(f"Time step: {t}")
        tot_it += k
        u1_s = u_s[:Nb]
        u2_s = u_s[Nb:]

        usol_f.append(u_f.copy())
        vsol_f.append(v_f.copy())
        psol_f.append(p_f.copy())

        usol_s.append(u_s.copy())
        vsol_s.append(v_s.copy())
        asol_s.append(a_s.copy())

    return usol_f, vsol_f, psol_f, usol_s, vsol_s, asol_s, P, T

if __name__ == "__main__":
    file_base = "inertia" 
    folder = "DATA"

    resolution = 0.04
    tol = 1e-7

    # FLUID domain
    lx_f, ly_f = 1.0, 1.0
    dx_f = resolution
    dy_f = resolution

    nx_f = np.floor(lx_f / dx_f).astype(int)
    ny_f = np.floor(ly_f / dy_f).astype(int)

     # Fluid material properties
    rho_f = 1.0              #density
    mu_f  = 0.001             #dynamic viscosity
    nu_f  = mu_f / rho_f     #kinematic viscosity
    print('nu_f', nu_f)
   
    vel0_f = [1.0, 0.0, 0.0, 0.0]  #Initial velocty [Ut, Ub, Vl, Vr]

    # SOLID domain
    lx_s = 0.5
    ly_s = 0.5

    dx_s = resolution 
    dy_s = resolution 

    nx_s = np.floor(lx_s / dx_s).astype(int)
    ny_s = np.floor(ly_s / dy_s).astype(int)
    Nb = (nx_s + 1) * (ny_s + 1)

    d0_x_s = 0.25 * np.ones(Nb)
    d0_y_s = 0.25 * np.ones(Nb)

    # Solid material properties
    E_s = 10000
    nu_s = 0.4
    rho_s = 1.5  # Solid density [kg/m^2]

    mu_s = E_s /(2*(1 + nu_s))
    lam_s =  E_s * nu_s /((1 + nu_s)* (1 - 2 * nu_s))
    
    dt = 0.01
    Nt = 1000

    # compute usol_f, vsol_f, psol_f and usol_s, vsol_s, asol_s
    u_f, v_f, p_f, u_s, v_s, a_s, P, T = solve(lx_f, ly_f, nx_f, ny_f, dx_f, dy_f, Nt, vel0_f, lx_s, ly_s, nx_s, ny_s, dx_s, dy_s, d0_x_s, d0_y_s, mu_s, lam_s, dt, Nb, nu_f, rho_f, rho_s, tol, minit=0, maxit=500)

    # ==========================
    # 1) SAVE PARAMETERS (CSV)
    # ==========================
    params = {
        "resolution": resolution,
        "lx_f": lx_f,
        "ly_f": ly_f,
        "nu_f": nu_f,
        "lx_s": lx_s,
        "ly_s": ly_s,
        "E_s": E_s,
        "nu_s": nu_s,
        "rho_s": rho_s,
        "rho_f": rho_f,
        "dt": dt,
        "Nt": Nt,
    }

    # Scrive CSV manualmente
    csv_file = f"{folder}/{file_base}.csv"
    npz_file = f"{folder}/{file_base}.npz"
    with open(csv_file, "w") as f:
        # Header
        f.write(",".join(params.keys()) + "\n")
        # Values
        f.write(",".join(str(v) for v in params.values()) + "\n")

    # ==========================
    # 2) SAVE NUMERICAL FIELDS (.npz)
    # ==========================
    np.savez(
        npz_file,

        # Fluid (time dependent)
        u_f=u_f,        
        v_f=v_f,
        p_f=p_f,

        # Solid (time dependent)
        u_s=u_s,
        v_s=v_s,
        a_s=a_s,

        # Mesh / geometry
        P=P,
        T=T,

        # Initial solid displacement
        d0_x_s=d0_x_s,
        d0_y_s=d0_y_s
    )

    print("✅ Saved")








    # print('uf', u_f[-1].shape)
    # print('vf', v_f[-1].shape)
    # print('pf', p_f[-1].shape)
    # print('us', u_s[-1].shape)
    # print('vs', v_s[-1].shape)

    # #  Visualization of the results
    # u_f_small = u_f[1:-1, 1:]
    # v_f_small = v_f[1:, 1:-1]
    # plt.quiver(u_f_small, v_f_small)
    # plt.quiver(u_f[-1], v_f[-1])
    # plt.show()

    # plt.quiver(u_s[-1], v_s[-1])
    # plt.show()

    # x_def = P[0, :] + u_s[-1][:Nb]
    # y_def = P[1, :] + u_s[-1][Nb]
    # for e in range(T.shape[1]):
    #     node_ids = T[:, e]
    #     #polygon_coords = [(x_def[i], y_def[i]) for i in node_ids]
    #     #polygon = Polygon(polygon_coords, closed=True, facecolor='black', edgecolor='black', linewidth=0.1)
    #     #ax.add_patch(polygon)
    #     x_coords = [x_def[i] for i in node_ids] + [x_def[node_ids[0]]] 
    #     y_coords = [y_def[i] for i in node_ids] + [y_def[node_ids[0]]]
    #     plt.plot(x_coords, y_coords, color='blue', linewidth=0.5)
    # plt.show()    

    # plt.quiver(u_f[-1], v_f[-1])
    # plt.show()

