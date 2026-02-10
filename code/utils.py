import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix, bmat, lil_matrix
from scipy.sparse.linalg import splu
from numpy.polynomial.legendre import leggauss

def solid_assembly(lx, ly, nx, ny, dx, dy, lam, mu, flag):
    """
    Assemble the finite element matrices for 2D linear elasticity
    on a structured quadrilateral mesh.

    The formulation corresponds to small-strain, linear, isotropic
    elasticity using Q1 (bilinear) finite elements.

    Parameters
    ----------
    lx, ly : float
        Length of the domain in x and y directions.
    nx, ny : int
        Number of elements in x and y directions.
    dx, dy : float
        Element size in x and y directions.
    lam, mu : float
        Lamé parameters (λ, μ).
    flag : array-like
        Boundary condition flags.
            shape: [bottom, right, top, left]
            flags: -2 for Neumann, -1 for Dirichlet

    Returns
    -------
    M_el : scipy.sparse matrix
        Block mass matrix for vector-valued displacement (2 DOFs per node).
    A : scipy.sparse matrix
        Global stiffness matrix of the elasticity problem.
    b : ndarray
        Global right-hand side vector.
    P : ndarray
        Coordinates of all mesh nodes, shape (2, Nb).
    T : ndarray
        Element connectivity matrix (4 nodes per element).
    M : scipy.sparse matrix
        Scalar mass matrix.
    bdry : ndarray
        Boundary description array.
    """

    # ------------------------------------------------------------------
    # Body force definitions (right-hand side)
    # ------------------------------------------------------------------
    # f1, f2 are the body forces in x- and y-directions.
    # Here they are set to zero everywhere.
    def f1(x, y):
        return x * 0

    def f2(x, y):
        return x * 0

    # ------------------------------------------------------------------
    # Mesh generation
    # ------------------------------------------------------------------
    # Create nodal coordinates
    x = np.linspace(0, lx, nx + 1)
    y = np.linspace(0, ly, ny + 1)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Stack coordinates into a 2 x Nb array
    P = np.vstack([X.ravel(), Y.ravel()])

    # ------------------------------------------------------------------
    # Element connectivity (Q1 bilinear elements)
    # ------------------------------------------------------------------
    # ix, jy index elements in x and y directions
    ix, jy = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')

    # Node numbering for each element (counter-clockwise)
    n1 = ix * (ny + 1) + jy
    n2 = n1 + (ny + 1)
    n3 = n2 + 1
    n4 = n1 + 1

    # Connectivity matrix: 4 x (nx*ny)
    T = np.stack([n1, n2, n3, n4], axis=-1).reshape(-1, 4).T

    # ------------------------------------------------------------------
    # Boundary description
    # [boundary flag, element index, node1, node2, side] 
    # ------------------------------------------------------------------
    bdry_list = []

    # Bottom boundary elements
    bdry_list += [
        [flag[0], el, T[0, el], T[1, el], 0]
        for el in range(0, nx * ny, ny)
    ]

    # Right boundary elements
    bdry_list += [
        [flag[1], el, T[1, el], T[2, el], 1]
        for el in range((nx - 1) * ny, nx * ny)
    ]

    # Top boundary elements
    bdry_list += [
        [flag[2], el, T[2, el], T[3, el], 2]
        for el in range(nx * ny - 1, -1, -ny)
    ]

    # Left boundary elements
    bdry_list += [
        [flag[3], el, T[3, el], T[0, el], 3]
        for el in range(ny - 1, -1, -1)
    ]

    bdry = np.array(bdry_list).T

    # ------------------------------------------------------------------
    # Body force evaluated at nodes
    # ------------------------------------------------------------------
    f1_v = f1(P[0, :], P[1, :])
    f2_v = f2(P[0, :], P[1, :])

    Nb = (nx + 1) * (ny + 1)  # number of nodes

    # ------------------------------------------------------------------
    # Local element matrices (Q1 elements)
    # ------------------------------------------------------------------
    # These matrices come from analytical integration of
    # bilinear basis functions on a rectangular element.

    # ∂/∂x contributions
    A_xx_l = (dy / dx) * np.array([
        [ 1/3, -1/3, -1/6,  1/6],
        [-1/3,  1/3,  1/6, -1/6],
        [-1/6,  1/6,  1/3, -1/3],
        [ 1/6, -1/6, -1/3,  1/3]
    ])

    # ∂/∂y contributions
    A_yy_l = (dx / dy) * np.array([
        [ 1/3,  1/6, -1/6, -1/3],
        [ 1/6,  1/3, -1/3, -1/6],
        [-1/6, -1/3,  1/3,  1/6],
        [-1/3, -1/6,  1/6,  1/3]
    ])

    # Mixed derivative contributions
    A_xy_l = np.array([
        [ 1/4, -1/4, -1/4,  1/4],
        [ 1/4, -1/4, -1/4,  1/4],
        [-1/4,  1/4,  1/4, -1/4],
        [-1/4,  1/4,  1/4, -1/4]
    ])

    # Consistent mass matrix
    M_l = (dx * dy) * np.array([
        [1/9,  1/18, 1/36, 1/18],
        [1/18, 1/9,  1/18, 1/36],
        [1/36, 1/18, 1/9,  1/18],
        [1/18, 1/36, 1/18, 1/9]
    ])

    # ------------------------------------------------------------------
    # Assembly preparation
    # ------------------------------------------------------------------
    Kg_xx = np.zeros((16, nx * ny))
    Kg_xy = np.zeros((16, nx * ny))
    Kg_yy = np.zeros((16, nx * ny))
    KM = np.zeros((16, nx * ny))

    # Local-to-global index mapping
    ii = np.array([0, 1, 2, 3] * 4)
    jj = np.array([0] * 4 + [1] * 4 + [2] * 4 + [3] * 4)

    Ig = T[ii, :]
    Jg = T[jj, :]

    # Fill element contributions
    for e in range(nx * ny):
        Kg_xx[:, e] = A_xx_l.flatten(order='F')
        Kg_xy[:, e] = A_xy_l.flatten(order='F')
        Kg_yy[:, e] = A_yy_l.flatten(order='F')
        KM[:, e] = M_l.flatten(order='F')

    # ------------------------------------------------------------------
    # Global sparse matrices
    # ------------------------------------------------------------------
    A_xx = coo_matrix((Kg_xx.flatten(), (Ig.flatten(), Jg.flatten())),
                      shape=(Nb, Nb))
    A_xy = coo_matrix((Kg_xy.flatten(), (Ig.flatten(), Jg.flatten())),
                      shape=(Nb, Nb))
    A_yy = coo_matrix((Kg_yy.flatten(), (Ig.flatten(), Jg.flatten())),
                      shape=(Nb, Nb))
    M = coo_matrix((KM.flatten(), (Ig.flatten(), Jg.flatten())),
                   shape=(Nb, Nb))

    # ------------------------------------------------------------------
    # Elasticity stiffness blocks
    # ------------------------------------------------------------------
    A11 = lam * A_xx + 2 * mu * A_xx + mu * A_yy
    A12 = lam * A_xy.T + mu * A_xy
    A21 = lam * A_xy + mu * A_xy.T
    A22 = lam * A_yy + 2 * mu * A_yy + mu * A_xx

    # Full block stiffness matrix
    A = bmat([[A11, A12],
              [A21, A22]], format='lil')

    # ------------------------------------------------------------------
    # Right-hand side
    # ------------------------------------------------------------------
    b = np.zeros(2 * Nb)
    b[:Nb] = M @ f1_v
    b[Nb:] = M @ f2_v

    # Block mass matrix for vector displacement
    M_el = sp.kron(sp.eye(2, format='csr'), M, format='csr')

    # ------------------------------------------------------------------
    # Dirichlet boundary conditions
    # ------------------------------------------------------------------
    mask = bdry[0] == -1
    dirichlet = np.unique(bdry[2:4, mask])

    for i in dirichlet:
        # x-displacement
        A[i, :] = 0.0
        A[i, i] = 1.0
        b[i] = 0.0

        # y-displacement
        A[i + Nb, :] = 0.0
        A[i + Nb, i + Nb] = 1.0
        b[i + Nb] = 0.0

    A = A.tocsc()

    return M_el, A, b, P, T, M, bdry

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import splu

def fluid_assembly_CF(nx, ny, dx, dy, G):
    """
    Assemble the linear system for the pressure Poisson equation
    used in the Chorin projection method for 2D incompressible Navier-Stokes.

    The discretization is a lowest-order finite volume method (equivalent
    to a centered finite difference) on a structured Cartesian mesh.

    Parameters
    ----------
    nx, ny : int
        Number of pressure control volumes in x and y directions.
    dx, dy : float
        Grid spacing in x and y directions.
    G : float
        Source term at the boundary (e.g., normal flux at inflow).

    Returns
    -------
    C : scipy.sparse.linalg.SuperLU
        LU factorization of the pressure matrix (ready for solving).
    b : ndarray
        Right-hand side vector including boundary contributions.

    Notes
    -----
    The system corresponds to solving the discrete Poisson equation:
    
        ∇² p = f

    on a 2D grid using a 5-point stencil for internal nodes.

    Boundary conditions:
    - Neumann (zero normal derivative) on South and North boundaries.
    - Dirichlet (fixed pressure) on West and East boundaries, with the West boundary having a source term G.
    """

    # Total number of pressure nodes
    nn = nx * ny

    # Initialize sparse matrix in LIL format for assembly
    A = lil_matrix((nn, nn))
    b = np.zeros(nn)

    # ---------------------------
    # Internal nodes (5-point stencil)
    # ---------------------------
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            # Local indices of current node and neighbors
            l = np.array([i, i+1, i-1, i, i])
            m = np.array([j, j, j, j+1, j-1])
            indices = np.ravel_multi_index([l, m], (nx, ny))

            # Fill matrix row for Laplacian
            A[indices[0], indices] = [-2*dx/dy - 2*dy/dx, dy/dx, dy/dx, dx/dy, dx/dy]

    # ---------------------------
    # Boundaries (South, North, West, East)
    # ---------------------------
    # South boundary (j=0)
    j = 0
    for i in range(1, nx - 1):
        l = np.array([i, i+1, i-1, i])
        m = np.array([j, j, j, j+1])
        indices = np.ravel_multi_index([l, m], (nx, ny))
        A[indices[0], indices] = [-dx/dy - 2*dy/dx, dy/dx, dy/dx, dx/dy]
        b[indices[0]] = 0

    # North boundary (j=ny-1)
    j = ny - 1
    for i in range(1, nx - 1):
        l = np.array([i, i+1, i-1, i])
        m = np.array([j, j, j, j-1])
        indices = np.ravel_multi_index([l, m], (nx, ny))
        A[indices[0], indices] = [-dx/dy - 2*dy/dx, dy/dx, dy/dx, dx/dy]
        b[indices[0]] = 0

    # West boundary (i=0)
    i = 0
    for j in range(1, ny - 1):
        l = np.array([i, i, i, i+1])
        m = np.array([j, j+1, j-1, j])
        indices = np.ravel_multi_index([l, m], (nx, ny))
        A[indices[0], indices] = [-dy/dx - 2*dx/dy, dx/dy, dx/dy, dy/dx]
        b[indices[0]] = G * dy

    # East boundary (i=nx-1)
    i = nx - 1
    for j in range(1, ny - 1):
        l = np.array([i, i, i, i-1])
        m = np.array([j, j+1, j-1, j])
        indices = np.ravel_multi_index([l, m], (nx, ny))
        A[indices[0], indices] = [-3*dy/dx - 2*dx/dy, dx/dy, dx/dy, dy/dx]
        b[indices[0]] = 0

    # ---------------------------
    # Corner nodes
    # ---------------------------
    # South-West corner
    i, j = 0, 0
    l = np.array([i, i+1, i])
    m = np.array([j, j, j+1])
    indices = np.ravel_multi_index([l, m], (nx, ny))
    A[indices[0], indices] = [-dx/dy - dy/dx, dy/dx, dx/dy]
    b[indices[0]] = G * dy

    # South-East corner
    i, j = nx - 1, 0
    l = np.array([i, i-1, i])
    m = np.array([j, j, j+1])
    indices = np.ravel_multi_index([l, m], (nx, ny))
    A[indices[0], indices] = [-dx/dy - 3*dy/dx, dy/dx, dx/dy]
    b[indices[0]] = 0

    # North-West corner
    i, j = 0, ny - 1
    l = np.array([i, i+1, i])
    m = np.array([j, j, j-1])
    indices = np.ravel_multi_index([l, m], (nx, ny))
    A[indices[0], indices] = [-dx/dy - dy/dx, dy/dx, dx/dy]
    b[indices[0]] = G * dy

    # North-East corner
    i, j = nx - 1, ny - 1
    l = np.array([i, i-1, i])
    m = np.array([j, j, j-1])
    indices = np.ravel_multi_index([l, m], (nx, ny))
    A[indices[0], indices] = [-dx/dy - 3*dy/dx, dy/dx, dx/dy]
    b[indices[0]] = 0

    # ---------------------------
    # Solve system using LU factorization
    # ---------------------------
    N = A.tocsc()
    C = splu(N)
    return C, b

import numpy as np

def fluid_solver_CF(u, v, nx, ny, dx, dy, dt, nu, Fx, Fy, Af, bf, vel0, rho):
    """
    Solve one timestep of the 2D incompressible Navier-Stokes equations
    using the Chorin projection method on on a staggered Marker-and-Cell (MAC) (see Figure) structured grid.

    The method is explicit in time for the convective and diffusive terms,
    and uses a pressure correction to enforce incompressibility.

    Parameters
    ----------
    u, v : ndarray
        Velocity fields in x and y directions (staggered grid / cell-centered).
    nx, ny : int
        Number of pressure nodes in x and y directions.
    dx, dy : float
        Grid spacing in x and y directions.
    dt : float
        Time step.
    nu : float
        Kinematic viscosity.
    Fx, Fy : ndarray
        External forces per unit mass (body forces).
    Af : SuperLU object
        LU factorization of the pressure Poisson matrix.
    bf : ndarray
        RHS vector contributions from boundaries for the pressure solve.
    vel0 : float
        Imposed velocity at the inflow boundary (left wall).
    rho : float
        Fluid density.

    Returns
    -------
    u, v : ndarray
        Updated velocity fields after pressure correction.
    p : ndarray
        Pressure field at the current timestep.

    Notes
    -----
    The algorithm consists of:
    1. Applying boundary conditions to the velocity.
    2. Computing intermediate velocities `ut`, `vt` (not divergence-free) using
       explicit convection and diffusion.
    3. Computing the divergence of the intermediate velocity.
    4. Solving the pressure Poisson equation to enforce incompressibility.
    5. Correcting the velocities using the pressure gradient.

    Boundary conditions:
    - Left wall: u = vel0, v = 0 (inlet prescribed velocity)
    - Right wall: Neumann (zero-gradient)
    - Top wall: no-slip (u = 0, v = 0)
    - Bottom wall: no-slip (u = 0, v = 0)
    """
    
    # Initialize intermediate velocities and pressure field
    p = np.zeros_like(u)
    ut = np.zeros_like(u)
    vt = np.zeros_like(v)
    divut = np.zeros_like(v)

    # ---------------------------
    # Boundary conditions
    # ---------------------------
    # Left wall: u = vel0, v = 0 (inlet prescribed velocity)
    u[1:-1,1] = vel0
    v[:,0] = -v[:,1]  # symmetry for wall

    # Right wall: Neumann (zero-gradient)
    u[:, -1] = u[:, -2]
    v[:, -1] = v[:, -2]

    # Top wall: no-slip
    u[-1,:] = - u[-2,:]
    v[-1,:] = 0.0

    # Bottom wall: no-slip
    u[0,:] = - u[1,:]
    v[1,:] = 0.0

    ut[1:-1, 1] = vel0  # inflow condition

    # ---------------------------
    # Compute intermediate u-velocity
    # ---------------------------
    # Average neighboring velocities (staggered to cell centers)
    ue = 0.5 * (u[1:ny+1, 3:nx+2] + u[1:ny+1, 2:nx+1])
    uw = 0.5 * (u[1:ny+1, 2:nx+1] + u[1:ny+1, 1:nx])
    un = 0.5 * (u[2:ny+2, 2:nx+1] + u[1:ny+1, 2:nx+1])
    us = 0.5 * (u[1:ny+1, 2:nx+1] + u[0:ny, 2:nx+1])
    vn = 0.5 * (v[2:ny+2, 1:nx] + v[2:ny+2, 2:nx+1])
    vs = 0.5 * (v[1:ny+1, 1:nx] + v[1:ny+1, 2:nx+1])

    # Convection term (explicit)
    convection = -(ue**2 - uw**2) / dx - (un * vn - us * vs) / dy
    # Diffusion term (explicit, central differences)
    diffusion = nu * ((u[1:ny+1, 1:nx] - 2.0 * u[1:ny+1, 2:nx+1] + u[1:ny+1, 3:nx+2]) / dx**2 +
                      (u[0:ny, 2:nx+1] - 2.0 * u[1:ny+1, 2:nx+1] + u[2:ny+2, 2:nx+1]) / dy**2)
    
    # Update intermediate u
    ut[1:ny+1, 2:nx+1] = u[1:ny+1, 2:nx+1] + dt * (convection + diffusion + Fx[:, 1:-1]/rho)

    # ---------------------------
    # Compute intermediate v-velocity
    # ---------------------------
    ve = 0.5 * (v[2:ny+1, 2:nx+2] + v[2:ny+1, 1:nx+1])
    vw = 0.5 * (v[2:ny+1, 1:nx+1] + v[2:ny+1, 0:nx])
    ue = 0.5 * (u[2:ny+1, 2:nx+2] + u[1:ny, 2:nx+2])
    uw = 0.5 * (u[2:ny+1, 1:nx+1] + u[1:ny, 1:nx+1])
    vn = 0.5 * (v[3:ny+2, 1:nx+1] + v[2:ny+1, 1:nx+1])
    vs = 0.5 * (v[2:ny+1, 1:nx+1] + v[1:ny, 1:nx+1])

    convection = -(ue * ve - uw * vw)/dx - (vn**2 - vs**2)/dy
    diffusion = nu * ((v[2:ny+1, 2:nx+2] - 2.0 * v[2:ny+1, 1:nx+1] + v[2:ny+1, 0:nx]) / dx**2 +
                      (v[3:ny+2, 1:nx+1] - 2.0 * v[2:ny+1, 1:-1] + v[1:ny, 1:nx+1]) / dy**2)
    
    vt[2:ny+1, 1:nx+1] = v[2:ny+1, 1:nx+1] + dt * (convection + diffusion + Fy[1:-1, :]/rho)

    # ---------------------------
    # Compute divergence of intermediate velocity
    # ---------------------------
    divut[1:-1, 1:-1] = (ut[1:-1, 2:] - ut[1:-1, 1:-1])/dx + (vt[2:, 1:-1] - vt[1:-1, 1:-1])/dy

    # Pressure RHS (from incompressibility)
    prhs = divut[1:-1, 1:-1].ravel('F') * dx * dy / dt

    # Solve pressure Poisson equation
    sol = Af.solve(prhs + bf)
    p[1:-1, 1:-1] = sol.reshape((ny, nx), order='F')

    # ---------------------------
    # Velocity correction
    # ---------------------------
    u[1:-1, 2:-1] = ut[1:-1, 2:-1] - dt * (p[1:-1, 2:-1] - p[1:-1, 1:-2])/dx
    v[2:-1, 1:-1] = vt[2:-1, 1:-1] - dt * (p[2:-1, 1:-1] - p[1:-2, 1:-1])/dy

    return u, v, p


"""
======================================================================================
Transfer Operators for Immersed Boundary (IB) / Immersed Domain (ID)
======================================================================================

In this section, we assemble the transfer operators used to couple the
fluid and solid meshes as explained in Section 3.3 of the paper. 

NB: Since we are using a MAC grid, the u and v velocity components live 
on different meshes. Therefore, for each solid point we must evaluate
in which x-cell and y-cell it falls, resulting in **two transfer operators**:

    - Tx: related to the x-component (applies to u-velocity and Lagrange multipliers)
    - Ty: related to the y-component (applies to v-velocity and Lagrange multipliers)

These operators are assembled by integrating the solid shape functions over
the fluid control volumes (using 2D quadrature) and then normalizing row sums.
"""

# ------------------------------------------------------------------------------------
# Section: ID (Immersed Domain)
# ------------------------------------------------------------------------------------
# Here we work on the full solid surface, resulting in 2D integrals
# to assemble Tx and Ty. The integrals are computed using Gaussian quadrature.

def fluid_cell_x(data, dx_f, dy_f, nx_f, ny_f):
    """
    Find the index of the x-velocity fluid cell in which a point falls.

    Parameters
    ----------
    data : array_like
        Coordinates of the solid point [x, y].
    dx_f, dy_f : float
        Grid spacing of the fluid mesh.
    nx_f, ny_f : int
        Number of cells in x and y directions for u-velocity.

    Returns
    -------
    fluid_index_x : int
        Flattened index of the fluid cell for x-component.
    """
    ix_x = np.floor((data[0] + dx_f/2) / dx_f).astype(int)
    iy_x = np.floor(data[1]/ dy_f).astype(int)

    ix_x = np.clip(ix_x, 0, nx_f)
    iy_x = np.clip(iy_x, 0, ny_f - 1)

    fluid_index_x = int(iy_x * (nx_f + 1) + ix_x)
    return fluid_index_x


def fluid_cell_y(data, dx_f, dy_f, nx_f, ny_f):
    """
    Find the index of the y-velocity fluid cell in which a point falls.

    Parameters
    ----------
    data : array_like
        Coordinates of the solid point [x, y].
    dx_f, dy_f : float
        Grid spacing of the fluid mesh.
    nx_f, ny_f : int
        Number of cells in x and y directions for v-velocity.

    Returns
    -------
    fluid_index_y : int
        Flattened index of the fluid cell for y-component.
    """
    ix_y = np.floor(data[0] / dx_f).astype(int)
    iy_y = np.floor((data[1] + dy_f/2)/ dy_f).astype(int)

    ix_y = np.clip(ix_y, 0, nx_f - 1)
    iy_y = np.clip(iy_y, 0, ny_f)

    fluid_index_y = int(iy_y * nx_f + ix_y)
    return fluid_index_y


def compute_quad(n):
    """
    Compute 2D Gaussian quadrature nodes and weights on [0,1]^2.

    Parameters
    ----------
    n : int
        Number of Gauss points per direction.

    Returns
    -------
    nodes_2d : list of tuples
        Coordinates of quadrature nodes.
    weights_2d : list of float
        Corresponding weights for each node.
    """
    a, b = 0, 1
    x_1, w_1 = leggauss(n)

    nodes_m = []
    weights_m = []
    for x, w in zip(x_1, w_1):
        x_mapped = (b - a)/2 * x + (a + b)/2
        w_mapped = (b - a)/2 * w
        nodes_m.append(x_mapped)
        weights_m.append(w_mapped)

    nodes_2d = [(x, y) for x in nodes_m for y in nodes_m]
    weights_2d = [wx * wy for wx in weights_m for wy in weights_m]
    return nodes_2d, weights_2d


def phi(i, x, y):
    """
    Bilinear shape function for a quadrilateral element.
    
    Parameters
    ----------
    i : int
        Local node index (0 to 3).
    x, y : float
        Local coordinates in reference element [0,1]^2.

    Returns
    -------
    value : float
        Shape function value at (x,y) for node i.
    """
    if i == 0: return (1.0 - x)*(1.0 - y)
    if i == 1: return x * (1.0 - y)
    if i == 2: return x * y
    if i == 3: return y * (1.0 - x)


def phi_x(x, y):
    """
    Derivative of bilinear shape functions with respect to x.
    Returns list of 4 derivatives for the 4 nodes.
    """
    return [(y - 1.0), (1.0 - y), y, -y]


def phi_y(x, y):
    """
    Derivative of bilinear shape functions with respect to y.
    Returns list of 4 derivatives for the 4 nodes.
    """
    return [(x - 1.0), -x, x, (1.0 - x)]


def compute_Jacobian(e, T, Pos, xq, yq):
    """
    Compute the Jacobian determinant for a quadrilateral element
    at a quadrature point.

    Parameters
    ----------
    e : int
        Element index.
    T : ndarray
        Connectivity matrix (4 nodes per element).
    Pos : ndarray
        Coordinates of solid nodes (2 x Nb).
    xq, yq : float
        Quadrature point coordinates in reference element.

    Returns
    -------
    detJ : float
        Absolute value of the Jacobian determinant.
    """
    phix = phi_x(xq, yq)
    phiy = phi_y(xq, yq)

    x_x = sum(phix[k] * Pos[0,T[k,e]] for k in range(4))
    x_y = sum(phiy[k] * Pos[0,T[k,e]] for k in range(4))
    y_x = sum(phix[k] * Pos[1,T[k,e]] for k in range(4))
    y_y = sum(phiy[k] * Pos[1,T[k,e]] for k in range(4))

    return np.abs(x_x * y_y - x_y * y_x)


def assembly_T_ID(Pos_x, Pos_y, dx_f, dy_f, nx_s, ny_s, nx_f, ny_f, T):
    """
    Assemble the transfer operators Tx and Ty for the full solid surface (ID).

    Parameters
    ----------
    Pos_x, Pos_y : ndarray
        Solid node coordinates.
    dx_f, dy_f : float
        Fluid grid spacing.
    nx_s, ny_s : int
        Solid mesh dimensions.
    nx_f, ny_f : int
        Fluid mesh dimensions.
    T : ndarray
        Connectivity matrix of solid elements.

    Returns
    -------
    Tx_normalized, Ty_normalized : ndarray
        Transfer operators for x- and y-components (normalized).
    """
    quad_nodes, quad_weights = compute_quad(4)
    Nb = (nx_s+1)*(ny_s+1)

    Tx = np.zeros((Nb, ny_f * (nx_f + 1)))
    Ty = np.zeros((Nb, (ny_f + 1) * nx_f))
    Pos = np.vstack((Pos_x, Pos_y))

    for e in range(T.shape[1]):
        for q, (xq, yq) in enumerate(quad_nodes):
            # Map quadrature node to physical coordinates
            xq_f = sum(phi(k, xq, yq) * Pos[:,T[k,e]] for k in range(4))

            # Identify fluid cells where x and y velocities live
            jx = fluid_cell_x(xq_f, dx_f, dy_f, nx_f, ny_f)
            jy = fluid_cell_y(xq_f, dx_f, dy_f, nx_f, ny_f)

            # Compute Jacobian
            J = compute_Jacobian(e, T, Pos, xq, yq)

            # Assemble Tx and Ty contributions
            for k in range(4):
                i = T[k,e]
                Tx[i, jx] += quad_weights[q] * phi(k, xq, yq) * J
                Ty[i, jy] += quad_weights[q] * phi(k, xq, yq) * J

    # Normalize rows
    row_sums_x = np.sum(Tx, axis=1)
    nonzero_rows_x = row_sums_x != 0
    Tx_normalized = Tx.astype(float).copy()
    Tx_normalized[nonzero_rows_x] /= row_sums_x[nonzero_rows_x, np.newaxis]

    row_sums_y = np.sum(Ty, axis=1)
    nonzero_rows_y = row_sums_y != 0
    Ty_normalized = Ty.astype(float).copy()
    Ty_normalized[nonzero_rows_y] /= row_sums_y[nonzero_rows_y, np.newaxis]

    return Tx_normalized, Ty_normalized


# ------------------------------------------------------------------------------------
# Section: IB (Immersed Boundary)
# ------------------------------------------------------------------------------------
# Here we work only on the solid boundary, resulting in 1D integrals
# along each boundary segment to assemble Tx and Ty. The integrals
# are computed using 1D Gaussian quadrature.

def phi_1d(k, x):
    """
    Linear 1D shape function for a boundary segment.

    Parameters
    ----------
    k : int
        Local node index (0 or 1) on the segment.
    x : float
        Local coordinate in reference segment [0,1].

    Returns
    -------
    value : float
        Shape function value at x for node k.
    """
    if k == 0:
        return 1 - x
    else:
        return x


def compute_quad_1d(n):
    """
    Compute 1D Gaussian quadrature nodes and weights on [0,1].

    Parameters
    ----------
    n : int
        Number of Gauss points.

    Returns
    -------
    nodes_m : list of float
        Quadrature nodes.
    weights_m : list of float
        Corresponding weights.
    """
    a, b = 0, 1
    x_1, w_1 = leggauss(n)
    nodes_m = []
    weights_m = []
    for x, w in zip(x_1, w_1):
        x_mapped = (b - a)/2 * x + (a + b)/2
        w_mapped = (b - a)/2 * w
        nodes_m.append(x_mapped)
        weights_m.append(w_mapped)
    return nodes_m, weights_m


def assembly_T_IB(Pos_x, Pos_y, dx_f, dy_f, nx_s, ny_s, nx_f, ny_f, bdry):
    """
    Assemble the IB transfer operators (Tx, Ty) by integrating along
    solid boundary segments using 1D Gaussian quadrature.

    Parameters
    ----------
    Pos_x, Pos_y : ndarray
        Coordinates of solid nodes.
    dx_f, dy_f : float
        Fluid grid spacing.
    nx_s, ny_s : int
        Solid mesh dimensions.
    nx_f, ny_f : int
        Fluid mesh dimensions.
    bdry : ndarray
        Boundary connectivity (node indices for each segment).

    Returns
    -------
    Tx_normalized, Ty_normalized : ndarray
        Transfer operators for x- and y-components (normalized).
    """
    # Quadrature nodes and weights for 1D segments
    quad_nodes, quad_weights = compute_quad_1d(4)

    Pos = np.vstack((Pos_x, Pos_y))
    Nb = (nx_s+1)*(ny_s+1)

    # Initialize operators
    Tx = np.zeros((Nb, ny_f * (nx_f + 1)))
    Ty = np.zeros((Nb, (ny_f + 1) * nx_f))

    # Loop over boundary segments
    for f in range(bdry.shape[1]):
        # Get coordinates of segment endpoints
        x1, y1 = Pos[:, bdry[2, f]]
        x2, y2 = Pos[:, bdry[3, f]]
        
        # Jacobian for linear segment (length)
        J = np.sqrt( (x2 - x1) ** 2 + (y2 - y1) ** 2 )
       
        # Loop over quadrature points
        for q in range(len(quad_weights)):
            xq_f = phi_1d(0, quad_nodes[q]) * Pos[:, bdry[2, f]] \
                 + phi_1d(1, quad_nodes[q]) * Pos[:, bdry[3, f]]
            
            # Identify fluid cells where x and y velocities live
            jx = fluid_cell_x(xq_f, dx_f, dy_f, nx_f, ny_f)
            jy = fluid_cell_y(xq_f, dx_f, dy_f, nx_f, ny_f)

            # Assemble contributions to Tx and Ty
            for idx in range(2):
                i = bdry[2 + idx, f]
                Tx[i, jx] += quad_weights[q] * phi_1d(idx, quad_nodes[q]) * J
                Ty[i, jy] += quad_weights[q] * phi_1d(idx, quad_nodes[q]) * J

    # Normalize rows
    row_sums_x = np.sum(Tx, axis=1)
    nonzero_rows_x = row_sums_x != 0
    Tx_normalized = Tx.astype(float).copy()
    Tx_normalized[nonzero_rows_x] /= row_sums_x[nonzero_rows_x, np.newaxis]

    row_sums_y = np.sum(Ty, axis=1)
    nonzero_rows_y = row_sums_y != 0
    Ty_normalized = Ty.astype(float).copy()
    Ty_normalized[nonzero_rows_y] /= row_sums_y[nonzero_rows_y, np.newaxis]

    return Tx_normalized, Ty_normalized


