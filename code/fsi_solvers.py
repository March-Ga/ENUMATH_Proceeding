import numpy as np
from scipy.sparse.linalg import factorized

from utils import *

# ------------------------------------------------------------------------------------
# Section: Fully Coupled Solver — Immersed Domain (ID)
# ------------------------------------------------------------------------------------
# This routine solves the partitioned fully coupled fluid–structure interaction
# problem using the Immersed Domain (ID) approach.
#
# The solid occupies a finite area inside the fluid domain and is treated
# through volume (2D) coupling integrals. The interaction between fluid and
# solid is enforced via Lagrange multipliers, transferred using the ID
# operators Tx and Ty.
#
# Time integration is performed using:
#   - Chorin projection method for the incompressible Navier–Stokes equations
#   - Newmark scheme for the solid dynamics
#
# At each time step, a fixed-point iteration is used to converge the coupled
# fluid–structure system.
def solve_ID(nx_f, ny_f, dx_f, dy_f, Nt, vel0_f,
             lx_s, ly_s, nx_s, ny_s, dx_s, dy_s,
             d0_x_s, d0_y_s, mu_s, lam_s, dt, Nb,
             nu_f, rho_f, G_f, rho_s,
             tol=1e-6, minit=0, maxit=100):
    """
    Fully coupled fluid–structure solver using the Immersed Domain (ID) method.

    - Fluid: incompressible Navier–Stokes equations (Chorin projection)
    - Solid: linear elasticity (Newmark time integration)
    - Coupling: volume-based transfer operators (Tx, Ty)
    """

    # --------------------------------------------------------------------------
    # Newmark parameters for solid time integration
    # --------------------------------------------------------------------------
    beta = 0.25
    gamma = 0.5
    
    # --------------------------------------------------------------------------
    # Fluid variables (MAC grid with ghost layers)
    # --------------------------------------------------------------------------
    p_f = np.zeros([ny_f + 2, nx_f + 2]) 
    u_f = np.zeros_like(p_f)
    v_f = np.zeros_like(p_f)

    # Store fluid solution history
    usol_f, vsol_f, psol_f = [], [], []
    usol_f.append(u_f.copy())
    vsol_f.append(v_f.copy())
    psol_f.append(p_f.copy())

    # --------------------------------------------------------------------------
    # Solid variables (displacement, velocity, acceleration)
    # --------------------------------------------------------------------------
    u_s = np.zeros(2 * Nb)   # displacement
    v_s = np.zeros(2 * Nb)   # velocity
    a_s = np.zeros(2 * Nb)   # acceleration

    # Store solid solution history
    usol_s, vsol_s, asol_s = [], [], []
    usol_s.append(u_s.copy())
    vsol_s.append(v_s.copy())
    asol_s.append(a_s.copy())

    # Lagrange multipliers enforcing coupling
    lam = np.zeros(2 * Nb)
    lam_old = np.zeros_like(lam)

    # --------------------------------------------------------------------------
    # Assemble solid matrices (mass, stiffness, mesh, boundary)
    # --------------------------------------------------------------------------
    M_el, A_s, b_s, P, T, M, bdry = solid_assembly(
        lx_s, ly_s, nx_s, ny_s, dx_s, dy_s,
        lam_s, mu_s, np.array([-2, -2, -2, -2])
    )

    # Assemble fluid pressure Poisson operator
    Af, bf = fluid_assembly_CF(nx_f, ny_f, dx_f, dy_f, G_f)

    tot_it = 0

    # ==========================================================================
    # Time-stepping loop
    # ==========================================================================
    for t in range(Nt):

        # ----------------------------------------------------------------------
        # Assemble ID transfer operators using current solid configuration
        # ----------------------------------------------------------------------
        Tx, Ty = assembly_T_ID(
            u_s[:Nb] + P[0, :] + d0_x_s,
            u_s[Nb:] + P[1, :] + d0_y_s,
            dx_f, dy_f, nx_s, ny_s, nx_f, ny_f, T
        )

        # Predictor step for Newmark scheme
        un = usol_s[t] + dt * vsol_s[t] + dt**2 * (0.5 - beta) * asol_s[t]
        vn = vsol_s[t] + dt * (1.0 - gamma) * asol_s[t]

        k = 0
        err = 1e5

        # ----------------------------------------------------------------------
        # Fixed-point iteration for fluid–structure coupling
        # ----------------------------------------------------------------------
        while (err > tol and k < maxit) or k < minit:

            # --------------------------------------------------------------
            # Interpolate fluid velocity to solid nodes (ID coupling)
            # --------------------------------------------------------------
            v_s_x = Tx @ u_f[1:-1, 1:].ravel()
            v_s_y = Ty @ v_f[1:, 1:-1].ravel()
            v_s = np.concatenate((v_s_x, v_s_y))

            # --------------------------------------------------------------
            # Update solid acceleration and displacement (Newmark)
            # --------------------------------------------------------------
            a_s = (v_s - vn) / (gamma * dt)
            u_s = un + beta * dt**2 * a_s

            # --------------------------------------------------------------
            # Compute Lagrange multipliers (coupling forces)
            # --------------------------------------------------------------
            lam = (rho_s - rho_f) * (M_el @ a_s) + (A_s @ u_s)

            # --------------------------------------------------------------
            # Spread solid forces back to fluid grid
            # --------------------------------------------------------------
            F_x = -Tx.T @ lam[:Nb]
            F_y = -Ty.T @ lam[Nb:]

            F_x = F_x.reshape((ny_f, nx_f + 1))
            F_y = F_y.reshape((ny_f + 1, nx_f))

            # --------------------------------------------------------------
            # Solve fluid problem with updated forcing
            # --------------------------------------------------------------
            u_f, v_f, p_f = fluid_solver_CF(
                usol_f[t].copy(), vsol_f[t].copy(),
                nx_f, ny_f, dx_f, dy_f, dt,
                nu_f, F_x, F_y, Af, bf, vel0_f, rho_f
            )

            # --------------------------------------------------------------
            # Convergence check
            # --------------------------------------------------------------
            err = np.linalg.norm(lam - lam_old, ord=2)
            lam_old = lam.copy()
            k += 1

        print(f'time step: {t}/{Nt}, iterations: {k}, error: {err}')
        tot_it += k

        # ----------------------------------------------------------------------
        # Store solutions
        # ----------------------------------------------------------------------
        usol_f.append(u_f.copy())
        vsol_f.append(v_f.copy())
        psol_f.append(p_f.copy())

        usol_s.append(u_s.copy())
        vsol_s.append(v_s.copy())
        asol_s.append(a_s.copy())

    return usol_f, vsol_f, psol_f, usol_s, vsol_s, asol_s, P, T, bdry, tot_it


# ------------------------------------------------------------------------------------
# Section: Fully Coupled Solver — Immersed Boundary (IB)
# ------------------------------------------------------------------------------------
# This routine solves the partitioned fully coupled fluid–structure interaction
# problem using the Immersed Boundary (IB) approach.
#
# In this case, the coupling is enforced only on the solid boundary, leading
# to line (1D) integrals along the interface. The interaction forces are
# transferred between solid and fluid through the IB transfer operators
# Tx and Ty.
#
# The solid dynamics are integrated using a Newmark scheme, while the fluid
# equations are advanced with a Chorin projection method. Dirichlet
# constraints are enforced on boundary degrees of freedom of the solid.
#
# A fixed-point iteration is performed at each time step to achieve
# convergence of the coupled system.

def solve_IB(nx_f, ny_f, dx_f, dy_f, Nt, vel0_f,
             lx_s, ly_s, nx_s, ny_s, dx_s, dy_s,
             d0_x_s, d0_y_s, mu_s, lam_s, dt, Nb,
             nu_f, rho_f, G_f, rho_s,
             tol=1e-6, minit=0, maxit=100):
    """
    Fully coupled fluid–structure solver using the Immersed Boundary (IB) method.

    - Fluid: incompressible Navier–Stokes equations (Chorin projection)
    - Solid: linear elasticity (Newmark time integration)
    - Coupling: boundary-based (1D) transfer operators Tx and Ty
    """

    # --------------------------------------------------------------------------
    # Newmark parameters for solid time integration
    # --------------------------------------------------------------------------
    beta = 0.25
    gamma = 0.5

    # --------------------------------------------------------------------------
    # Fluid variables (MAC grid with ghost layers)
    # --------------------------------------------------------------------------
    p_f = np.zeros([ny_f + 2, nx_f + 2])
    u_f = np.zeros_like(p_f)
    v_f = np.zeros_like(p_f)

    # Store fluid solution history
    usol_f, vsol_f, psol_f = [], [], []
    usol_f.append(u_f.copy())
    vsol_f.append(v_f.copy())
    psol_f.append(p_f.copy())

    # --------------------------------------------------------------------------
    # Solid variables (displacement, velocity, acceleration)
    # --------------------------------------------------------------------------
    u_s = np.zeros(2 * Nb)
    v_s = np.zeros(2 * Nb)
    a_s = np.zeros(2 * Nb)

    # Store solid solution history
    usol_s, vsol_s, asol_s = [], [], []
    usol_s.append(u_s.copy())
    vsol_s.append(v_s.copy())
    asol_s.append(a_s.copy())

    # Lagrange multipliers enforcing fluid–solid coupling
    lam = np.zeros(2 * Nb)
    lam_old = np.zeros_like(lam)

    # --------------------------------------------------------------------------
    # Assemble solid matrices and boundary information
    # --------------------------------------------------------------------------
    M_el, A_s, b_s, P, T, M, bdry = solid_assembly(
        lx_s, ly_s, nx_s, ny_s, dx_s, dy_s,
        lam_s, mu_s, np.array([-2, -2, -2, -2])
    )

    # --------------------------------------------------------------------------
    # Build solid system matrix for IB formulation
    # (includes inertia + stiffness, time-discretized)
    # --------------------------------------------------------------------------
    R = (rho_s - rho_f) * M_el / (gamma * dt) + (beta * dt / gamma) * A_s

    # Solid boundary nodes (Dirichlet constraints)
    dirichlet = np.unique(bdry[2:4, :])

    # Enforce Dirichlet conditions on solid boundary DOFs
    for i in dirichlet:
        R[i, :] = 0.0
        R[i, i] = 1.0
        R[i + Nb, :] = 0.0
        R[i + Nb, i + Nb] = 1.0

    # Factorize solid operator once (efficiency)
    R = R.tocsc()
    solve_s = factorized(R)

    # Assemble fluid pressure Poisson operator
    Af, bf = fluid_assembly_CF(nx_f, ny_f, dx_f, dy_f, G_f)

    tot_it = 0

    # ==========================================================================
    # Time-stepping loop
    # ==========================================================================
    for t in range(Nt):

        # ----------------------------------------------------------------------
        # Assemble IB transfer operators (boundary integrals)
        # ----------------------------------------------------------------------
        Tx, Ty = assembly_T_IB(
            u_s[:Nb] + P[0, :] + d0_x_s,
            u_s[Nb:] + P[1, :] + d0_y_s,
            dx_f, dy_f, nx_s, ny_s, nx_f, ny_f, bdry
        )

        # Predictor step (Newmark scheme)
        un = usol_s[t] + dt * vsol_s[t] + dt**2 * (0.5 - beta) * asol_s[t]
        vn = vsol_s[t] + dt * (1.0 - gamma) * asol_s[t]

        # Right-hand side for solid velocity solve
        rhs = (rho_s - rho_f) * (1.0 / (gamma * dt)) * M_el @ vn \
              - A_s @ (un - (beta * dt / gamma) * vn)

        k = 0
        err = 1e5

        # ----------------------------------------------------------------------
        # Fixed-point iteration for fluid–structure coupling
        # ----------------------------------------------------------------------
        while (err > tol and k < maxit) or k < minit:

            # --------------------------------------------------------------
            # Interpolate fluid velocity to solid boundary
            # --------------------------------------------------------------
            v_s_x = Tx @ u_f[1:-1, 1:].ravel()
            v_s_y = Ty @ v_f[1:, 1:-1].ravel()

            # Enforce boundary velocities on solid DOFs
            for i in dirichlet:
                rhs[i] = v_s_x[i]
                rhs[i + Nb] = v_s_y[i]

            # --------------------------------------------------------------
            # Solve for solid velocity
            # --------------------------------------------------------------
            v_s = solve_s(rhs)

            # --------------------------------------------------------------
            # Update solid acceleration and displacement
            # --------------------------------------------------------------
            a_s = (v_s - vn) / (gamma * dt)
            u_s = un + beta * dt**2 * a_s

            # --------------------------------------------------------------
            # Compute Lagrange multipliers (interface forces)
            # --------------------------------------------------------------
            lam = (rho_s - rho_f) * (M_el @ a_s) + (A_s @ u_s)

            # --------------------------------------------------------------
            # Spread interface forces back to fluid grid
            # --------------------------------------------------------------
            F_x = -Tx.T @ lam[:Nb]
            F_y = -Ty.T @ lam[Nb:]

            F_x = F_x.reshape((ny_f, nx_f + 1))
            F_y = F_y.reshape((ny_f + 1, nx_f))

            # --------------------------------------------------------------
            # Solve fluid problem with IB forcing
            # --------------------------------------------------------------
            u_f, v_f, p_f = fluid_solver_CF(
                usol_f[t].copy(), vsol_f[t].copy(),
                nx_f, ny_f, dx_f, dy_f, dt,
                nu_f, F_x, F_y, Af, bf, vel0_f, rho_f
            )

            # --------------------------------------------------------------
            # Convergence check
            # --------------------------------------------------------------
            err = np.linalg.norm(lam - lam_old, ord=2)
            lam_old = lam.copy()
            k += 1

        print(f'time step: {t}/{Nt}, iterations: {k}, error: {err}')
        tot_it += k

        # ----------------------------------------------------------------------
        # Store solutions
        # ----------------------------------------------------------------------
        usol_f.append(u_f.copy())
        vsol_f.append(v_f.copy())
        psol_f.append(p_f.copy())

        usol_s.append(u_s.copy())
        vsol_s.append(v_s.copy())
        asol_s.append(a_s.copy())

    return usol_f, vsol_f, psol_f, usol_s, vsol_s, asol_s, P, T, bdry, tot_it
