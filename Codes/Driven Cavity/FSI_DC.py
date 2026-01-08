import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import bmat
import scipy.sparse as sp
from scipy.sparse.linalg import factorized
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import splu
from numpy.polynomial.legendre import leggauss
from shapely.geometry import Polygon, box
from shapely.geometry import MultiPoint

def solid_assembly(lx, ly, nx, ny, dx, dy, lam, mu, dt):
    def f1(x,y):
        return x*0

    def f2(x,y):
        return x*0

    x = np.linspace(0, lx, nx+1)
    y = np.linspace(0, ly, ny+1)
    X, Y = np.meshgrid(x, y, indexing='ij')
    P = np.vstack([X.ravel(), Y.ravel()])

    ix, jy = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    n1 = ix * (ny + 1) + jy
    n2 = n1 + (ny + 1)
    n3 = n2 + 1
    n4 = n1 + 1
    T = np.stack([n1, n2, n3, n4], axis=-1).reshape(-1, 4).T

    flag = np.array([-2, -2, -2, -2])       # bottom, right, top, left
    bdry_list = []

    # bottom boundary
    bdry_list += [
        [flag[0], el, T[0, el], T[1, el], 0]
        for el in range(0, nx * ny, ny)]

    # right boundary
    bdry_list += [
        [flag[1], el, T[1, el], T[2, el], 1]
        for el in range((nx-1) * (ny), nx * ny)]

    # top boundary
    bdry_list += [
        [flag[2], el, T[2, el], T[3, el], 2]
        for el in range(nx * ny - 1, -1, -ny)]

    # left boundary
    bdry_list += [
        [flag[3], el, T[3, el], T[0, el], 3]
        for el in range(ny - 1, -1, -1)]

    bdry = np.array(bdry_list).T

    f1_v = f1(P[0,:], P[1,:])
    f2_v = f2(P[0,:], P[1,:])


    Nb = (nx+1)*(ny+1)

    A_xx_l = (dy/dx) * np.array([[ 1/3, -1/3, -1/6, 1/6], [-1/3, 1/3, 1/6, -1/6], [-1/6, 1/6, 1/3, -1/3], [1/6, -1/6, -1/3, 1/3] ])
    A_yy_l = (dx/dy) * np.array([[ 1/3, 1/6, -1/6, -1/3], [1/6, 1/3, -1/3, -1/6], [-1/6, -1/3, 1/3, 1/6], [-1/3, -1/6, 1/6, 1/3] ])
    A_xy_l = np.array([[ 1/4, -1/4, -1/4, 1/4], [1/4, -1/4, -1/4, 1/4], [-1/4, 1/4, 1/4, -1/4], [-1/4, 1/4, 1/4, -1/4] ])
    M_l =(dx*dy) * np.array([[ 1/9, 1/18, 1/36, 1/18], [1/18, 1/9, 1/18, 1/36], [1/36, 1/18, 1/9, 1/18], [1/18, 1/36, 1/18, 1/9]])

    Kg_xx = np.zeros((16, nx*ny))
    Kg_xy = np.zeros((16, nx*ny))
    Kg_yy = np.zeros((16, nx*ny))
    KM = np.zeros((16, nx*ny))

    ii = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])
    jj = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3 ,3 ,3])

    Ig = T[ii, :]
    Jg = T[jj, :]

    for e in range(nx*ny):
        Kg_xx[:, e] = A_xx_l.flatten(order='F')
        Kg_xy[:, e] = A_xy_l.flatten(order='F')
        Kg_yy[:, e] = A_yy_l.flatten(order='F')
        KM[:, e] = M_l.flatten(order='F')

    A_xx = coo_matrix((Kg_xx.flatten(), (Ig.flatten(), Jg.flatten())), shape=(Nb, Nb))
    A_xy = coo_matrix((Kg_xy.flatten(), (Ig.flatten(), Jg.flatten())), shape=(Nb, Nb))
    A_yy = coo_matrix((Kg_yy.flatten(), (Ig.flatten(), Jg.flatten())), shape=(Nb, Nb))
    M = coo_matrix((KM.flatten(), (Ig.flatten(), Jg.flatten())), shape=(Nb, Nb))

    A11 = lam * A_xx + 2 * mu * A_xx + mu * A_yy
    A12 = lam * A_xy.T + mu * A_xy
    A21 = lam * A_xy + mu * A_xy.T
    A22 = lam * A_yy + 2 * mu * A_yy + mu * A_xx

    A = bmat([[A11, A12],
            [A21, A22]], format='lil')

    b = np.zeros(2*Nb)
    b[:Nb] = M * f1_v
    b[Nb:] = M * f2_v
    

    M_el = sp.kron(sp.eye(2, format='csr'), M, format='csr')

#    # Neumann BC
#    def g1(i, x):
#        if i == 'b':
#            return 0
#        elif i == 'r':
#            return 0
#        elif i == 't':
#            return 0
#        elif i == 'l':
#            return 0
#        
#    def g2(i, x):
#        if i == 'b':
#            return 0
#        elif i == 'r':
#            return 0
#        elif i == 't':
#            return 0
#        elif i == 'l':
#            return 0
#    
#    def phi(i, x, y):
#        if i == 0:
#            return (1.0 - x)*(1.0 - y)
#        if i == 1:
#            return x * (1.0 - y)
#        if i == 2:
#            return x * y
#        if i == 3:
#            return y * (1.0 - x)


#    v_1 = np.zeros((Nb))
#    v_2 = np.zeros((Nb))
#
#    nbn = bdry.shape[1]
#    for k in range(nbn):
#        if bdry[0,k] == -2:
#            el = bdry[1,k]
#            for i in range(4):
#                if bdry[4,k] == 0:
#                    r1 = dx * (g1('b', dx * 1/3 + P[0, bdry[2,k]]) * phi(i, 1/3, 0) + g1('b', dx * 2/3 + P[0, bdry[2,k]]) * phi(i, 2/3, 0))
#                    r2 = dx * (g2('b', dx * 1/3 + P[0, bdry[2,k]]) * phi(i, 1/3, 0) + g2('b', dx * 2/3 + P[0, bdry[2,k]]) * phi(i, 2/3, 0))
#                elif bdry[4,k] == 1:
#                    r1 = dy * (g1('r', dy * 1/3 + P[1, bdry[2,k]]) * phi(i, 1, 1/3) + g1('r', dy * 2/3 + P[1, bdry[2,k]]) * phi(i, 1, 2/3))
#                    r2 = dy * (g2('r', dy * 1/3 + P[1, bdry[2,k]]) * phi(i, 1, 1/3) + g2('r', dy * 2/3 + P[1, bdry[2,k]]) * phi(i, 1, 2/3))
#                elif bdry[4,k] == 2:
#                    r1 = dx * (g1('t', dx * 1/3 + P[0, bdry[3,k]]) * phi(i, 1/3, 1) + g1('t', dx * 2/3 + P[0, bdry[3,k]]) * phi(i, 2/3, 1))
#                    r2 = dx * (g2('t', dx * 1/3 + P[0, bdry[3,k]]) * phi(i, 1/3, 1) + g2('t', dx * 2/3 + P[0, bdry[3,k]]) * phi(i, 2/3, 1))
#                elif bdry[4,k] == 3:
#                    #print(bdry[2,k], "-",bdry[3,k])
#                    r1 = dy * (g1('l', dy * 1/3 + P[1, bdry[3,k]]) * phi(i, 0, 1/3) + g1('l', dy * 2/3 + P[1, bdry[3,k]]) * phi(i, 0, 2/3))
#                    r2 = dy * (g2('l', dy * 1/3 + P[1, bdry[3,k]]) * phi(i, 0, 1/3) + g2('l', dy * 2/3 + P[1, bdry[3,k]]) * phi(i, 0, 2/3))
#
#                v_1[T[i, el]] = v_1[T[i, el]] + r1
#                v_2[T[i, el]] = v_2[T[i, el]] + r2
#
#
#    
#    b[:Nb] = b[:Nb] + v_1
#    b[Nb:] = b[Nb:] + v_2
    
    mask = bdry[0] == -1
    dirichlet = np.unique(bdry[2:4, mask])
    
    for k in range(len(dirichlet)):
        i = dirichlet[k]
        #i = bdry[2,k]
        A[i, :] = 0.0
        A[i, i] = 1.0
        b[i] = 0.0
        A[i + Nb, :] = 0.0
        A[i + Nb, i + Nb] = 1.0
        b[i + Nb] = 0.0
    A = A.tocsc()

    beta = 0.25
    gamma = 0.5

    R = M_el/(gamma * dt) + (beta * dt / gamma) * A
    #R =(beta * dt / gamma) * A
    R = R.tocsc()
    #solve = factorized(R)

    return R, M_el, A, b, P, T, M, bdry



def fluid_assembly(nx, ny, dx, dy):
    nn = nx * ny
    A = lil_matrix((nn, nn))
    b = np.zeros(nn)

    # Punti interni
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            l = np.array([i, i+1, i-1, i, i])
            m = np.array([j, j, j, j+1, j-1])
            indices = np.ravel_multi_index([l, m], (nx, ny))
            A[indices[0], indices] = [-2*dx/dy - 2*dy/dx, dy/dx, dy/dx, dx/dy, dx/dy]
            

    # bordo sud
    j = 0
    for i in range(1, nx - 1):
        l = np.array([i, i+1, i-1, i])
        m = np.array([j, j, j, j+1])
        indices = np.ravel_multi_index([l, m], (nx, ny))
        A[indices[0], indices] = [-dx/dy - 2*dy/dx, dy/dx, dy/dx, dx/dy]
        b[indices[0]] = 0

    # bordo nord
    j = ny - 1
    for i in range(1, nx - 1):
        l = np.array([i, i+1, i-1, i])
        m = np.array([j, j, j, j-1])
        indices = np.ravel_multi_index([l, m], (nx, ny))
        A[indices[0], indices] = [-dx/dy - 2*dy/dx, dy/dx, dy/dx, dx/dy]
        b[indices[0]] = 0

    # bordo ovest
    i = 0
    for j in range(1, ny - 1):
        l = np.array([i, i, i, i+1])
        m = np.array([j, j+1, j-1, j])
        indices = np.ravel_multi_index([l, m], (nx, ny))
        A[indices[0], indices] = [-dy/dx - 2*dx/dy, dx/dy, dx/dy, dy/dx]
        b[indices[0]] = 0

    # bordo est
    i = nx - 1
    for j in range(1, ny - 1):
        l = np.array([i, i, i, i-1])
        m = np.array([j, j+1, j-1, j])
        indices = np.ravel_multi_index([l, m], (nx, ny))
        A[indices[0], indices] = [-dy/dx - 2*dx/dy, dx/dy, dx/dy, dy/dx]
        b[indices[0]] = 0

    # angolo sud-ovest
    i = 0
    j = 0
    l = np.array([i, i+1, i])
    m = np.array([j, j, j+1])
    indices = np.ravel_multi_index([l, m], (nx, ny))
    A[indices[0], indices] = [-dx/dy -dy/dx, dy/dx, dx/dy]
    b[indices[0]] = 0

    # angolo sud-est
    i = nx - 1
    j = 0
    l = np.array([i, i-1, i])
    m = np.array([j, j, j+1])
    indices = np.ravel_multi_index([l, m], (nx, ny))
    A[indices[0], indices] = [-dx/dy - dy/dx, dy/dx, dx/dy]
    b[indices[0]] = 0

    # angolo nord-ovest
    i = 0
    j = ny - 1
    l = np.array([i, i+1, i])
    m = np.array([j, j, j-1])
    indices = np.ravel_multi_index([l, m], (nx, ny))
    A[indices[0], indices] = [-dx/dy - dy/dx, dy/dx, dx/dy]
    b[indices[0]] =  0

    # angolo nord-est
    i = nx - 1
    j = ny - 1
    l = np.array([i, i-1, i])
    m = np.array([j, j, j-1])
    indices = np.ravel_multi_index([l, m], (nx, ny))
    A[indices[0], indices] = [-dx/dy - dy/dx, dy/dx, dx/dy]
    b[indices[0]] = 0

    N = A.tocsc()
    C = splu(N)
    return C, b

# Fx e Fy fanno riferimento solo a celle attive, che significa che escludo le celle 100% esterne
# faccio lo stesso nel calcolo di Bx e By, quindi le dimensioni tornano tutte

def fluid_solver(u, v, nx, ny, dx, dy, dt, nu, Fx, Fy, C, vel0):
    p = np.zeros_like(u)
    ut = np.zeros_like(u)
    vt = np.zeros_like(v)
    divut = np.zeros_like(v)
    
    Ut, Ub, Vl, Vr = vel0
    
    # Left wall
    u[:,1] = 0.0
    v[:,0] = 2.0 * Vl - v[:,1]

    # Right wall
    u[:,-1] = 0.0
    v[:, -1] = 2.0 * Vr - v[:,-2]

    # Top wall
    u[-1,:] = 2.0 * Ut - u[-2,:]
    v[-1,:] = 0.0

    # Bottom wall
    u[0,:] = 2.0 * Ub - u[1,:]
    v[1,:] = 0.0
    
    
    ue = 0.5 * (u[1:ny+1, 3:nx+2] + u[1:ny+1, 2:nx+1])
    uw = 0.5 * (u[1:ny+1, 2:nx+1] + u[1:ny+1, 1:nx])
    un = 0.5 * (u[2:ny+2, 2:nx+1] + u[1:ny+1, 2:nx+1])
    us = 0.5 * (u[1:ny+1, 2:nx+1] + u[0:ny, 2:nx+1])
    vn = 0.5 * (v[2:ny+2, 1:nx] + v[2:ny+2, 2:nx+1])
    vs = 0.5 * (v[1:ny+1, 1:nx] + v[1:ny+1, 2:nx+1])
    
    convection = -(ue**2 - uw**2) / dx - (un * vn - us * vs) / dy
    diffusion = nu * ((u[1:ny+1, 1:nx] - 2.0 * u[1:ny+1, 2:nx+1] + u[1:ny+1, 3:nx+2]) / dx**2 +
                    (u[0:ny, 2:nx+1] - 2.0 * u[1:ny+1, 2:nx+1] + u[2:ny+2, 2:nx+1]) / dy**2)
    
    ut[1:ny+1, 2:nx+1] = u[1:ny+1, 2:nx+1] + dt * (convection + diffusion) + Fx[:, 1:-1]
    
    ve = 0.5 * (v[2:ny+1, 2:nx+2] + v[2:ny+1, 1:nx+1])
    vw = 0.5 * (v[2:ny+1, 1:nx+1] + v[2:ny+1, 0:nx])
    ue = 0.5 * (u[2:ny+1, 2:nx+2] + u[1:ny, 2:nx+2])
    uw = 0.5 * (u[2:ny+1, 1:nx+1] + u[1:ny, 1:nx+1])
    vn = 0.5 * (v[3:ny+2, 1:nx+1] + v[2:ny+1, 1:nx+1])
    vs = 0.5 * (v[2:ny+1, 1:nx+1] + v[1:ny, 1:nx+1])
    convection = -(ue * ve - uw * vw)/dx - (vn**2 - vs**2)/dy
    diffusion = nu * ((v[2:ny+1, 2:nx+2] - 2.0 * v[2:ny+1, 1:nx+1] + v[2:ny+1, 0:nx]) / dx**2 +
                    (v[3:ny+2, 1:nx+1] - 2.0 * v[2:ny+1, 1:nx+1] + v[1:ny, 1:nx+1]) / dy**2)
 
    vt[2:ny+1, 1: nx+1] = v[2:ny+1, 1: nx+1] + dt*(convection + diffusion) + Fy[1:-1, :]
    # Perchè Fy[1:ny, 0:nx] e non Fy[0:ny, 0:nx]

    divut[1:-1, 1:-1] = (ut[1:-1, 2:] - ut[1:-1, 1:-1])/dx + (vt[2:, 1:-1] - vt[1:-1, 1:-1])/dy
    prhs = divut[1:-1,1:-1].ravel('F') * dx * dy / dt
    
    sol = C.solve(prhs)
    
    p[1:-1, 1:-1] = sol.reshape((ny, nx), order='F')

    u[1:-1, 2:-1] = ut[1:-1, 2:-1] - dt * (p[1:-1, 2:-1] - p[1:-1, 1:-2])/dx     # solo interior, bc già fatte
    v[2:-1, 1:-1] = vt[2:-1, 1:-1] - dt * (p[2:-1, 1:-1] - p[1:-2, 1:-1])/dy

    return u, v, p


def phi(i, x, y):
    if i == 0:
        return (1.0 - x)*(1.0 - y)
    if i == 1:
        return x * (1.0 - y)
    if i == 2:
        return x * y
    if i == 3:
        return y * (1.0 - x)
    
def phi_x(x, y):
    return [(y - 1.0), (1.0 - y), y, -y]
    
def phi_y(x, y):
    return [(x - 1.0), -x, x, (1.0 - x)]

    
def compute_Jacobian(e, T, Pos, xq, yq):
    phix = phi_x(xq, yq)
    phiy = phi_y(xq, yq)
    
    x_x = phix[0] * Pos[0,T[0,e]] + phix[1] * Pos[0,T[1,e]] + phix[2] * Pos[0,T[2,e]] + phix[3] * Pos[0,T[3,e]]
    x_y = phiy[0] * Pos[0,T[0,e]] + phiy[1] * Pos[0,T[1,e]] + phiy[2] * Pos[0,T[2,e]] + phiy[3] * Pos[0,T[3,e]]

    y_x = phix[0] * Pos[1,T[0,e]] + phix[1] * Pos[1,T[1,e]] + phix[2] * Pos[1,T[2,e]] + phix[3] * Pos[1,T[3,e]]
    y_y = phiy[0] * Pos[1,T[0,e]] + phiy[1] * Pos[1,T[1,e]] + phiy[2] * Pos[1,T[2,e]] + phiy[3] * Pos[1,T[3,e]]

    return np.abs(x_x * y_y - x_y * y_x)

def fluid_cell_x(data, dx_f, dy_f, nx_f, ny_f):
    ix_x = np.floor((data[0] + dx_f/2) / dx_f).astype(int)
    iy_x = np.floor(data[1]/ dy_f).astype(int)

    ix_x = np.clip(ix_x, 0, nx_f)
    iy_x = np.clip(iy_x, 0, ny_f - 1)

    fluid_index_x = int(iy_x * (nx_f + 1) + ix_x)
    return fluid_index_x
    
def fluid_cell_y(data, dx_f, dy_f, nx_f, ny_f):
    ix_y = np.floor(data[0] / dx_f).astype(int)
    iy_y = np.floor((data[1] + dy_f/2)/ dy_f).astype(int)

    ix_y = np.clip(ix_y, 0, nx_f - 1)
    iy_y = np.clip(iy_y, 0, ny_f)

    fluid_index_y = int(iy_y * nx_f + ix_y)
    return fluid_index_y

def compute_quad(n):
    n_digits = 20
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
    
def assembly_B_ID(Pos_x, Pos_y, dx_f, dy_f, nx_s, ny_s, nx_f, ny_f, T):
    quad_nodes, quad_weights = compute_quad(4)

    Nb = (nx_s+1)*(ny_s+1)
    Bx = np.zeros((Nb, ny_f * (nx_f + 1)))
    By = np.zeros((Nb, (ny_f + 1) * nx_f))
    Pos = np.vstack((Pos_x, Pos_y))

    for e in range(T.shape[1]):
        for q in range(len(quad_nodes)):
            xq_f = phi(0,quad_nodes[q][0], quad_nodes[q][1]) * Pos[:,T[0,e]] + phi(1,quad_nodes[q][0], quad_nodes[q][1]) * Pos[:,T[1,e]] + phi(2,quad_nodes[q][0], quad_nodes[q][1]) * Pos[:,T[2,e]] + phi(3,quad_nodes[q][0], quad_nodes[q][1]) * Pos[:,T[3,e]]
            
            jx = fluid_cell_x(xq_f, dx_f, dy_f, nx_f, ny_f)
            jy = fluid_cell_y(xq_f, dx_f, dy_f, nx_f, ny_f)

            J = compute_Jacobian(e, T, Pos, quad_nodes[q][0], quad_nodes[q][1])

            for k in range(4):
                i = T[k,e]
                Bx[i, jx] += quad_weights[q] * phi(k, quad_nodes[q][0], quad_nodes[q][1]) * J
                By[i, jy] += quad_weights[q] * phi(k, quad_nodes[q][0], quad_nodes[q][1]) * J

    row_sums_x = np.sum(Bx, axis=1)
    nonzero_rows_x = row_sums_x != 0
    Bx_normalized = Bx.astype(float).copy()
    Bx_normalized[nonzero_rows_x] = Bx_normalized[nonzero_rows_x] / row_sums_x[nonzero_rows_x, np.newaxis]

#    Bx_normalized = np.where(row_sums_x[:, np.newaxis] != 0, Bx / row_sums_x[:, np.newaxis], 0.0)
    
    row_sums_y = np.sum(By, axis=1)
    nonzero_rows_y = row_sums_y != 0
    By_normalized = By.astype(float).copy()
    By_normalized[nonzero_rows_y] = By_normalized[nonzero_rows_y] / row_sums_y[nonzero_rows_y, np.newaxis]
    
#    By_normalized = np.where(row_sums_y[:, np.newaxis] != 0, By / row_sums_y[:, np.newaxis], 0.0)
    return Bx_normalized, By_normalized


def assembly_B_ID_Pen(Pos_x, Pos_y, dx_f, dy_f, nx_s, ny_s, nx_f, ny_f, T):
    quad_nodes, quad_weights = compute_quad(4)

    Nb = (nx_s+1)*(ny_s+1)
#    Bx = np.zeros((ny_f * (nx_f + 1), Nb))
#    By = np.zeros(((ny_f + 1) * nx_f, Nb))
    Bx = np.zeros((Nb, ny_f * (nx_f + 1)))
    By = np.zeros((Nb, (ny_f + 1) * nx_f))
    Pos = np.vstack((Pos_x, Pos_y))

    for e in range(T.shape[1]):
        for q in range(len(quad_nodes)):
            xq_f = phi(0,quad_nodes[q][0], quad_nodes[q][1]) * Pos[:,T[0,e]] + phi(1,quad_nodes[q][0], quad_nodes[q][1]) * Pos[:,T[1,e]] + phi(2,quad_nodes[q][0], quad_nodes[q][1]) * Pos[:,T[2,e]] + phi(3,quad_nodes[q][0], quad_nodes[q][1]) * Pos[:,T[3,e]]
            
            jx = fluid_cell_x(xq_f, dx_f, dy_f, nx_f, ny_f)
            jy = fluid_cell_y(xq_f, dx_f, dy_f, nx_f, ny_f)

            J = compute_Jacobian(e, T, Pos, quad_nodes[q][0], quad_nodes[q][1])

            for k in range(4):
                i = T[k,e]
                Bx[i, jx] += quad_weights[q] * phi(k, quad_nodes[q][0], quad_nodes[q][1]) * J
                By[i, jy] += quad_weights[q] * phi(k, quad_nodes[q][0], quad_nodes[q][1]) * J

    return Bx, By


def fluid_solver_Pen(u, v, nx, ny, dx, dy, dt, nu, vf_x, vf_y, C, vel0, mu, Ix, Iy):
    p = np.zeros_like(u)
    ut = np.zeros_like(u)
    vt = np.zeros_like(v)
    divut = np.zeros_like(v)
    
    Ut, Ub, Vl, Vr = vel0
    
    # Left wall
    u[:,1] = 0.0
    v[:,0] = 2.0 * Vl - v[:,1]

    # Right wall
    u[:,-1] = 0.0
    v[:, -1] = 2.0 * Vr - v[:,-2]

    # Top wall
    u[-1,:] = 2.0 * Ut - u[-2,:]
    v[-1,:] = 0.0

    # Bottom wall
    u[0,:] = 2.0 * Ub - u[1,:]
    v[1,:] = 0.0
    
    
    ue = 0.5 * (u[1:ny+1, 3:nx+2] + u[1:ny+1, 2:nx+1])
    uw = 0.5 * (u[1:ny+1, 2:nx+1] + u[1:ny+1, 1:nx])
    un = 0.5 * (u[2:ny+2, 2:nx+1] + u[1:ny+1, 2:nx+1])
    us = 0.5 * (u[1:ny+1, 2:nx+1] + u[0:ny, 2:nx+1])
    vn = 0.5 * (v[2:ny+2, 1:nx] + v[2:ny+2, 2:nx+1])
    vs = 0.5 * (v[1:ny+1, 1:nx] + v[1:ny+1, 2:nx+1])
    
    convection = -(ue**2 - uw**2) / dx - (un * vn - us * vs) / dy
    diffusion = nu * ((u[1:ny+1, 1:nx] - 2.0 * u[1:ny+1, 2:nx+1] + u[1:ny+1, 3:nx+2]) / dx**2 +
                    (u[0:ny, 2:nx+1] - 2.0 * u[1:ny+1, 2:nx+1] + u[2:ny+2, 2:nx+1]) / dy**2)
  

    ut[1:ny+1, 2:nx+1] = u[1:ny+1, 2:nx+1] * (1 + Ix[:,1:-1] * mu) + dt * (convection + diffusion) + Ix[:,1:-1] * mu * vf_x[:,1:-1]
    
    ve = 0.5 * (v[2:ny+1, 2:nx+2] + v[2:ny+1, 1:nx+1])
    vw = 0.5 * (v[2:ny+1, 1:nx+1] + v[2:ny+1, 0:nx])
    ue = 0.5 * (u[2:ny+1, 2:nx+2] + u[1:ny, 2:nx+2])
    uw = 0.5 * (u[2:ny+1, 1:nx+1] + u[1:ny, 1:nx+1])
    vn = 0.5 * (v[3:ny+2, 1:nx+1] + v[2:ny+1, 1:nx+1])
    vs = 0.5 * (v[2:ny+1, 1:nx+1] + v[1:ny, 1:nx+1])
    convection = -(ue * ve - uw * vw)/dx - (vn**2 - vs**2)/dy
    diffusion = nu * ((v[2:ny+1, 2:nx+2] - 2.0 * v[2:ny+1, 1:nx+1] + v[2:ny+1, 0:nx]) / dx**2 +
                    (v[3:ny+2, 1:nx+1] - 2.0 * v[2:ny+1, 1:nx+1] + v[1:ny, 1:nx+1]) / dy**2)
 
    vt[2:ny+1, 1: nx+1] = v[2:ny+1, 1: nx+1] * (1 + Iy[1:-1,:] * mu) + dt*(convection + diffusion) + Iy[1:-1,:] * mu * vf_y[1:-1,:]
    # Perchè Fy[1:ny, 0:nx] e non Fy[0:ny, 0:nx]

    divut[1:-1, 1:-1] = (ut[1:-1, 2:] - ut[1:-1, 1:-1])/dx + (vt[2:, 1:-1] - vt[1:-1, 1:-1])/dy
    prhs = divut[1:-1,1:-1].ravel('F') * dx * dy / dt
    
    sol = C.solve(prhs)
    
    p[1:-1, 1:-1] = sol.reshape((ny, nx), order='F')

    u[1:-1, 2:-1] = ut[1:-1, 2:-1] - dt * (p[1:-1, 2:-1] - p[1:-1, 1:-2])/dx     # solo interior, bc già fatte
    v[2:-1, 1:-1] = vt[2:-1, 1:-1] - dt * (p[2:-1, 1:-1] - p[1:-2, 1:-1])/dy

    return u, v, p


def assembly_indicator(Pos_x, Pos_y, dx_f, dy_f, nx_s, ny_s, nx_f, ny_f, lx_f, ly_f):
    cell_area = dx_f * dy_f
    
    points = np.column_stack((Pos_x, Pos_y))
    polygon = MultiPoint(points).convex_hull
    
    Bx = np.zeros((nx_f + 1, ny_f))
    By = np.zeros((nx_f, ny_f + 1))

    X_x = np.linspace(-dx_f/2, lx_f + dx_f/2, nx_f + 2)
    X_y = np.linspace(0, ly_f, ny_f + 1)

    Y_x = np.linspace(0, lx_f, nx_f + 1)
    Y_y = np.linspace(-dy_f,ly_f + dy_f/2, ny_f + 2)

    for i in range(nx_f + 1):
        for j in range(ny_f):
            cell = box(X_x[i], X_y[j], X_x[i+1], X_y[j+1])
            inter_area = polygon.intersection(cell).area
            Bx[i, j] = inter_area / cell_area

    for i in range(nx_f):
        for j in range(ny_f + 1):
            cell = box(Y_x[i], Y_y[j], Y_x[i+1], Y_y[j+1])
            inter_area = polygon.intersection(cell).area
            By[i, j] = inter_area / cell_area

    return Bx, By
