from multiprocessing.resource_sharer import stop
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import splu
import matplotlib.pyplot as plt
import os

from FSI_DC import assembly_B_ID_Pen
from FSI_DC import assembly_indicator

# # TO DO
#
# - validate using reference solution, as did before - DONE
# - compute at each time, the residual of full NS (divergence should be ok)
# - fai interpolazione fine - DONE
# 
#
# # # # 


def assemble_lapl(nx, ny, dx, dy):
    nn = nx * ny
    A = lil_matrix((nn, nn))
    b = np.zeros(nn)
    #S = S[1:-1,1:-1]
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
    b[indices[0]] = 0

    # angolo nord-est
    i = nx - 1
    j = ny - 1
    l = np.array([i, i-1, i])
    m = np.array([j, j, j-1])
    indices = np.ravel_multi_index([l, m], (nx, ny))
    A[indices[0], indices] = [-dx/dy -dy/dx, dy/dx, dx/dy]
    b[indices[0]] = 0

    #M = A.tocsr()
    N = A.tocsc()
    C = splu(N)

    return C, b

def solve(lx, ly, nx, ny, Nt, vel0, mu):


    lx_s = 0.15
    ly_s = 0.15
    nx_s = 5
    ny_s = 5
    Nb = (nx_s + 1) * (ny_s + 1)

    x = np.linspace(0, lx_s, nx_s +1)
    y = np.linspace(0, ly_s, ny_s + 1)
    X, Y = np.meshgrid(x, y, indexing='ij')
    P = np.vstack([X.ravel(), Y.ravel()])

    ix, jy = np.meshgrid(np.arange(nx_s), np.arange(ny_s), indexing='ij')
    n1 = ix * (ny_s + 1) + jy
    n2 = n1 + (ny_s + 1)
    n3 = n2 + 1
    n4 = n1 + 1
    T = np.stack([n1, n2, n3, n4], axis=-1).reshape(-1, 4).T

    d0x, d0y = 0.275, 0.25

    x_obj = P[0, :] + d0x 
    y_obj = P[1, :] + d0y 

    dx = lx / nx
    dy = ly / ny

    p = np.zeros([ny + 2, nx + 2])      # N.B. i e j sono invertiti!!!!

    u = np.zeros_like(p)
    ut = np.zeros_like(p) # velocità u tilde
    v = np.zeros_like(p)
    vt = np.zeros_like(p) # velocità v tilde
    divut = np.zeros_like(p)  # divergenza di u tilde, mi serve per il passaggio dove calcolo laplaciano pressione

    Ut, Ub, Vl, Vr = vel0

    usol=[]
    usol.append(u.copy())

    vsol=[]
    vsol.append(v.copy())

    psol = []
    psol.append(p.copy())

    A_lapl, _ = assemble_lapl(nx, ny, dx, dy)
    Bx, By = assembly_B_ID_Pen(x_obj, y_obj, dx, dy, nx_s, ny_s, nx, ny, T) 
    Ix, Iy = assembly_indicator(x_obj, y_obj, dx, dy, nx_s, ny_s, nx, ny, lx, ly) 
    Ix = Ix.T
    Iy = Iy.T
    print('Ix')
    print(Ix)

    vs_x = np.zeros(Nb)
    vs_y = np.zeros(Nb)

    vf_x = Bx.T @ vs_x
    vf_y = By.T @ vs_y
    # print('velocities')
    # print(max(abs(vf_x)))
    # print(max(abs(vf_y)))

    vf_x = vf_x.reshape((ny, nx + 1))  
    vf_y = vf_y.reshape((ny + 1, nx))

    for _ in range(0, Nt):

        # Boundary conditions, all'inizio per conservare la massa
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

        ut[1:ny+1, 2:nx+1] = u[1:ny+1, 2:nx+1] + dt * (convection + diffusion) + mu * Ix[:,1:-1] * (vf_x[:,1:-1] + u[1:ny+1, 2:nx+1])
        # print(mu * Ix[:,1:-1] * (vf_x[:,1:-1] + u[1:ny+1, 2:nx+1]))dt * mu * Ix[:,1:-1] * (vf_x[:,1:-1] -
        # print(u[1:ny+1, 2:nx+1])
        # print(u[1:ny+1, 2:nx+1] * (Ix[:,1:-1] * mu))


        ve = 0.5 * (v[2:ny+1, 2:nx+2] + v[2:ny+1, 1:nx+1])
        vw = 0.5 * (v[2:ny+1, 1:nx+1] + v[2:ny+1, 0:nx])
        ue = 0.5 * (u[2:ny+1, 2:nx+2] + u[1:ny, 2:nx+2])
        uw = 0.5 * (u[2:ny+1, 1:nx+1] + u[1:ny, 1:nx+1])
        vn = 0.5 * (v[3:ny+2, 1:nx+1] + v[2:ny+1, 1:nx+1])
        vs = 0.5 * (v[2:ny+1, 1:nx+1] + v[1:ny, 1:nx+1])
        convection = -(ue * ve - uw * vw)/dx - (vn**2 - vs**2)/dy
        diffusion = nu * ((v[2:ny+1, 2:nx+2] - 2.0 * v[2:ny+1, 1:nx+1] + v[2:ny+1, 0:nx]) / dx**2 +
                        (v[3:ny+2, 1:nx+1] - 2.0 * v[2:ny+1, 1:nx+1] + v[1:ny, 1:nx+1]) / dy**2)

        vt[2:ny+1, 1: nx+1] = v[2:ny+1, 1: nx+1] + dt*(convection + diffusion) + mu * Iy[1:-1,:] * (vf_y[1:-1,:] + v[2:ny+1, 1: nx+1])
    # dt * mu * Iy[1:-1,:] * (vf_y[1:-1,:] -
        divut[1:-1, 1:-1] = (ut[1:-1, 2:] - ut[1:-1, 1:-1])/dx + (vt[2:, 1:-1] - vt[1:-1, 1:-1])/dy   # solo interior points
        prhs = divut[1:-1,1:-1].ravel('F') * dx * dy / dt

        sol = A_lapl.solve(prhs)
        p[1:-1, 1:-1] = sol.reshape((ny, nx), order='F')

        #a = A_lapl.solve(prhs)
        #p = np.zeros([ny + 2, nx + 2]) 
        #p[1:-1,1:-1]=a.reshape([1:-1,1:-1].shape, order='F')

        u[1:-1, 2:-1] = ut[1:-1, 2:-1] - dt * (p[1:-1, 2:-1] - p[1:-1, 1:-2])/dx     # solo interior, bc già fatte
        v[2:-1, 1:-1] = vt[2:-1, 1:-1] - dt * (p[2:-1, 1:-1] - p[1:-2, 1:-1])/dy     # solo interior, bc già fatte

        usol.append(u.copy())
        vsol.append(v.copy())
        psol.append(p.copy())


    return usol, vsol, psol

def interpolation(usol, vsol, psol, fine = False):
    """
    Fine interpolation: at each pressure volume, we have 4 velocity subdomains, 
    so we can divide each control volume in 4 equal subvolumes, each ones 
    with a values of u and v depending on the corresponding u and v volume
    """
    u_int = []
    v_int = []
    p_int = []
    if fine:
        u_fine = np.zeros((2*nx, 2*ny))
        v_fine = np.zeros((2*nx, 2*ny))
        for t in range(len(usol)):
            u = usol[t]
            v = vsol[t]
            p = psol[t]
            for i in range(nx):
                for j in range(ny):
                    u_left = u[i,j]       
                    u_right = u[i,j+1]    
                    u_fine[2*i:2*i+2, 2*j]   = u_left
                    u_fine[2*i:2*i+2, 2*j+1] = u_right
                    v_bottom = v[i,j]     
                    v_top = v[i+1,j]   
                    v_fine[2*i, 2*j:2*j+2]   = v_bottom
                    v_fine[2*i+1, 2*j:2*j+2] = v_top

            p_int.append(p[1:-1, 1:-1]) 
            u_int.append(u_fine)
            v_int.append(v_fine)

    else:
        for t in range(len(usol)):
            u = usol[t]
            v = vsol[t]
            p = psol[t]
            
            p_int.append(p[1:-1, 1:-1]) 
            u_int.append((u[1:-1, 2:] + u[1:-1, 1:-1]) * 0.5)
            v_int.append((v[2: , 1:-1] + v[1:-1 , 1:-1]) * 0.5)
    return u_int, v_int, p_int

def plot_divergence(u, v, nx, ny, dx, dy):
    div = np.zeros((nx, ny))
    div = (u[1:-1, 2:] - u[1:-1, 1:-1])/dx + (v[2: , 1:-1] - v[1:-1 , 1:-1])/dy
    plt.imshow(div, origin="lower")
    plt.title('Divergence of u')
    plt.colorbar()
    plt.show()
    
def visual(usol, vsol, psol, lx, nx, dx, ly, ny, dy):
    plt.quiver(usol[-1], vsol[-1])
    plt.show()

    # speed = np.sqrt(usol[-1]*usol[-1] + vsol[-1]*vsol[-1])
    # plt.contourf(speed)
    # plt.show()

    # x = np.linspace(dx/2, lx - dx/2, nx)
    # y = np.linspace(dy/2, ly - dy/2, ny)

    
    # xx,yy = np.meshgrid(x,y)
    # fig = plt.figure(figsize=[3,3],dpi=200)
    # plt.quiver(xx, yy, usol[-1], vsol[-1])
    # plt.xlim([xx[0,0],xx[0,-1]])
    # plt.ylim([yy[0,0],yy[-1,0]])
    # plt.streamplot(xx, yy, usol[-1], vsol[-1], color = usol[-1], density=1.5, cmap=plt.cm.autumn, linewidth=1.0)
    # plt.colorbar()
    # plt.gca().set_aspect('auto')
    # plt.show()

    # plt.pcolormesh(xx, yy, psol[-1], shading='nearest', cmap='Blues')
    # plt.colorbar()
    # plt.show()


def create_video(u_i, v_i, p_i, name, lx, nx, dx, ly, ny, dy):
    dir_path = os.path.join("video", name)
    os.makedirs(dir_path, exist_ok=True)

    x = np.linspace(dx/2, lx - dx/2, nx)
    y = np.linspace(dy/2, ly - dy/2, ny)

    speed_min = 1.0
    speed_max = 0.0

    xx, yy = np.meshgrid(x,y)
    j=0
    for i in range(Nt):
        if (i % 10 ==0):
            fig = plt.figure(figsize=[7,4],dpi=200)
            u = u_i[i]
            v = v_i[i]
            speed = np.sqrt(u*u + v*v)

            fig, ax = plt.subplots(figsize=[4, 4], dpi=200)
            plt.contourf(xx, yy,speed)#, levels=50, cmap='viridis', vmin=speed_min, vmax=speed_max)
            plt.streamplot(xx,yy,u, v, color=speed, density=1.5, cmap=plt.cm.autumn, linewidth=1.0)
            plt.ylim(yy.min(), yy.max())
            plt.xlim(xx.min(), xx.max())
            
            frame_path = os.path.join(dir_path, f"frame_{j:04d}.png")
            plt.savefig(frame_path)
            j = j+1
            plt.close()


if __name__ == "__main__":
     # Domain
    lx, ly = 1.0, 1.0
    mu = 1e10

    # Number of volumes per direction (pressure volumes)
    nx, ny = 50, 50
    dt = 0.01
    Nt = 5 

    nu = 0.01

    #Initial velocty [Ut, Ub, Vl, Vr]
    vel0 = [1.0, 0.0, 0.0, 0.0]


    dx = lx/nx
    dy = ly/ny
    usol, vsol, psol = solve(lx, ly, nx, ny, Nt, vel0, mu)


    # plot_divergence(usol[-1], vsol[-1], nx, ny, lx/nx, ly/ny)
    u_i, v_i, p_i = interpolation(usol, vsol, psol, False)

    visual(u_i, v_i, p_i, lx, nx, dx, ly, ny, dy)

    x = np.linspace(0, lx, nx)
    y = np.linspace(0, ly, ny)
    xx, yy = np.meshgrid(x,y)

    lx_s = 0.15
    ly_s = 0.15
    nx_s = 5
    ny_s = 5
    Nb = (nx_s + 1) * (ny_s + 1)

    x = np.linspace(0, lx_s, nx_s +1)
    y = np.linspace(0, ly_s, ny_s + 1)
    X, Y = np.meshgrid(x, y, indexing='ij')
    P = np.vstack([X.ravel(), Y.ravel()])

    ix, jy = np.meshgrid(np.arange(nx_s), np.arange(ny_s), indexing='ij')
    n1 = ix * (ny_s + 1) + jy
    n2 = n1 + (ny_s + 1)
    n3 = n2 + 1
    n4 = n1 + 1
    T = np.stack([n1, n2, n3, n4], axis=-1).reshape(-1, 4).T

    d0x, d0y = 0.275, 0.25

    x_obj = P[0, :] + d0x 
    y_obj = P[1, :] + d0y


    x_def = x_obj
    y_def = y_obj
    plt.streamplot(xx, yy, u_i[-1], v_i[-1], color=u_i[-1], density=1.5, cmap=plt.cm.autumn, linewidth=1.0)
    for e in range(T.shape[1]):
        node_ids = T[:, e]
    
        x_coords = [x_def[i] for i in node_ids] + [x_def[node_ids[0]]] 
        y_coords = [y_def[i] for i in node_ids] + [y_def[node_ids[0]]]
        plt.plot(x_coords, y_coords, color='blue', linewidth=0.5)

    plt.xlim(0, lx)
    plt.ylim(0, ly)
    plt.show()
    # create_video(u_i, v_i, p_i, 'test0', lx, nx, dx, ly, ny, dy)

    # base = np.linspace(0,1,nx)
    # u = u_i[-1]

    # plt.figure(figsize=(5.5, 4)) 
    # plt.scatter(u[:, int(nx/2)], base, s=10, c='blue', marker='x') 
    # plt.plot(u[:, int(nx/2)], base, color='red', linestyle='-', linewidth=1) 
    # plt.xlim(-0.4, 1.2)  
    # plt.ylim(0.0, 1.0)  
    # plt.show()
