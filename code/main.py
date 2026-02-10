from fsi_solvers import *   
import os         

# =============================================================================
# Main driver script for Fluid–Structure Interaction (FSI) simulations
# using either:
#   - IB : Immersed Boundary method  (boundary coupling)
#   - ID : Immersed Domain method    (volume coupling)
#
# The script:
#   1. Defines geometry, meshes, and material parameters
#   2. Builds initial conditions
#   3. Runs the coupled solver
#   4. Saves results and simulation parameters to disk
# =============================================================================
if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # OUTPUT SETTINGS
    # -------------------------------------------------------------------------
    file_base = "test"              # Base filename for saved results
    os.makedirs("DATA", exist_ok=True)  # Create output folder if missing
    folder = "DATA"                 # Directory for results

    # -------------------------------------------------------------------------
    # SELECT COUPLING METHOD
    # -------------------------------------------------------------------------
    IB = True       # Immersed Boundary method (boundary integrals)
    ID = False      # Immersed Domain method (volume integrals)

    # -------------------------------------------------------------------------
    # NUMERICAL PARAMETERS
    # -------------------------------------------------------------------------
    resolution = 0.02   # Grid spacing for both fluid and solid meshes
    tol = 1e-7          # Fixed-point iteration tolerance for FSI coupling

    # ========================================================================
    # FLUID DOMAIN DEFINITION
    # ========================================================================
    lx_f, ly_f = 5.0, 1.0     # Fluid domain length in x and y directions

    dx_f = resolution         # Fluid grid spacing in x
    dy_f = resolution         # Fluid grid spacing in y

    # Number of fluid cells
    nx_f = np.floor(lx_f / dx_f).astype(int)
    ny_f = np.floor(ly_f / dy_f).astype(int)

    # -------------------------------------------------------------------------
    # FLUID MATERIAL PROPERTIES
    # -------------------------------------------------------------------------
    rho_f = 1.0      # Fluid density [kg/m^3]
    mu_f  = 0.01     # Dynamic viscosity [Pa·s]
    nu_f  = mu_f / rho_f   # Kinematic viscosity [m^2/s]
    G_f = 0.0        # inlet pressure gradient

    # -------------------------------------------------------------------------
    # INITIAL FLUID VELOCITY PROFILE
    # Example: Poiseuille flow (parabolic velocity profile)
    # -------------------------------------------------------------------------
    U_max = 1.0
    y = np.linspace(dy_f/2, ly_f - dy_f/2, ny_f)
    vel0_f = U_max * (1 - ((y - ly_f / 2) / (ly_f / 2))**2)

    # ========================================================================
    # SOLID DOMAIN DEFINITION
    # ========================================================================
    lx_s = 0.5      # Solid width
    ly_s = 0.5      # Solid height

    dx_s = resolution   # Solid grid spacing in x
    dy_s = resolution   # Solid grid spacing in y

    # Number of solid elements
    nx_s = np.floor(lx_s / dx_s).astype(int)
    ny_s = np.floor(ly_s / dy_s).astype(int)

    # Number of solid nodes
    Nb = (nx_s + 1) * (ny_s + 1)

    # -------------------------------------------------------------------------
    # INITIAL SOLID POSITION
    # These represent rigid offsets of the solid in the fluid domain
    # -------------------------------------------------------------------------
    d0_x_s = 1.5 * np.ones(Nb)   # Initial x-displacement
    d0_y_s = 0.25 * np.ones(Nb)  # Initial y-displacement

    # -------------------------------------------------------------------------
    # SOLID MATERIAL PROPERTIES (linear elasticity)
    # -------------------------------------------------------------------------
    E_s = 1e6     # Young’s modulus [Pa]
    nu_s = 0.4    # Poisson ratio
    rho_s = 100   # Solid density [kg/m^2]

    # Lamé parameters
    mu_s = E_s /(2*(1 + nu_s))                 # Shear modulus
    lam_s = E_s * nu_s /((1 + nu_s)*(1 - 2*nu_s))  # First Lamé parameter

    # ========================================================================
    # TIME DISCRETIZATION
    # ========================================================================
    dt = 1e-3               # Time step size
    T_final = 0.5           # Final simulation time
    Nt = int(T_final/dt)    # Number of time steps


    if IB:
        u_f, v_f, p_f, u_s, v_s, a_s, P, T, B, max_it = solve_IB(
            nx_f, ny_f, dx_f, dy_f, Nt, vel0_f,
            lx_s, ly_s, nx_s, ny_s, dx_s, dy_s,
            d0_x_s, d0_y_s,
            mu_s, lam_s,
            dt, Nb,
            nu_f, rho_f, G_f, rho_s,
            tol, minit=0, maxit=500
        )

    elif ID:
        u_f, v_f, p_f, u_s, v_s, a_s, P, T, B, max_it = solve_ID(
            nx_f, ny_f, dx_f, dy_f, Nt, vel0_f,
            lx_s, ly_s, nx_s, ny_s, dx_s, dy_s,
            d0_x_s, d0_y_s,
            mu_s, lam_s,
            dt, Nb,
            nu_f, rho_f, G_f, rho_s,
            tol, minit=0, maxit=500
        )

    else:
        raise ValueError("Please set either IB or ID to True.")

    # ========================================================================
    # SAVE SIMULATION PARAMETERS (CSV)
    # ========================================================================
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

    csv_file = f"{folder}/{file_base}.csv"
    npz_file = f"{folder}/{file_base}.npz"

    # Write CSV manually
    with open(csv_file, "w") as f:
        f.write(",".join(params.keys()) + "\n")
        f.write(",".join(str(v) for v in params.values()) + "\n")

    # ========================================================================
    # SAVE NUMERICAL RESULTS
    # ========================================================================
    np.savez(
        npz_file,

        # Fluid fields (time-dependent)
        u_f=u_f,
        v_f=v_f,
        p_f=p_f,

        # Solid fields (time-dependent)
        u_s=u_s,
        v_s=v_s,
        a_s=a_s,

        # Solid mesh
        P=P,   # node coordinates
        T=T,   # element connectivity
        B=B,   # boundary nodes

        # Initial solid position
        d0_x_s=d0_x_s,
        d0_y_s=d0_y_s
    )

    print("Saved")
