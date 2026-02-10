from fsi_solvers import *

if __name__ == "__main__":
    file_base = "IB_TEST_0" 
    folder = "TEST 1/DATA_T1"

    resolution = 0.02
    tol = 1e-7

    # FLUID domain
    lx_f, ly_f = 5.0, 1.0
    dx_f = resolution
    dy_f = resolution

    nx_f = np.floor(lx_f / dx_f).astype(int)
    ny_f = np.floor(ly_f / dy_f).astype(int)

     # Fluid material properties
    rho_f = 1.0             # density
    mu_f  = 0.01              # dynamic viscosity
    nu_f  = mu_f / rho_f     # kinematic viscosity
    G_f = 0.0
   
    U_max = 1.0
    y = np.linspace(dy_f/2, ly_f - dy_f/2, ny_f)
    vel0_f = U_max * (1 - ((y - ly_f / 2) / (ly_f / 2))**2)

    # SOLID domain
    lx_s = 0.5
    ly_s = 0.5

    dx_s = resolution
    dy_s = resolution

    nx_s = np.floor(lx_s / dx_s).astype(int)
    ny_s = np.floor(ly_s / dy_s).astype(int)
    Nb = (nx_s + 1) * (ny_s + 1)

    # Initial solid displacement
    d0_x_s = 1.5 * np.ones(Nb)
    d0_y_s = 0.25 * np.ones(Nb)

    # Solid material properties
    E_s = 1e6
    nu_s = 0.4
    rho_s = 100   # Solid density [kg/m^2]

    mu_s = E_s /(2*(1 + nu_s))
    lam_s =  E_s * nu_s /((1 + nu_s)* (1 - 2 * nu_s))
   
    
    dt = 1e-3
    Nt = int(0.5/dt)
    print(Nt)
    

    # compute usol_f, vsol_f, psol_f and usol_s, vsol_s, asol_s
    u_f, v_f, p_f, u_s, v_s, a_s, P, T, B, max_it = solve_ID(nx_f, ny_f, dx_f, dy_f, Nt, vel0_f, lx_s, ly_s, nx_s, ny_s, dx_s, dy_s, d0_x_s, d0_y_s, mu_s, lam_s, dt, Nb, nu_f, rho_f, G_f, rho_s, tol, minit=0, maxit=500)

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
        B=B,

        # Initial solid displacement
        d0_x_s=d0_x_s,
        d0_y_s=d0_y_s
    )

    print("✅ Saved")