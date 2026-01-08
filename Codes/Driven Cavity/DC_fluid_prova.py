import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import splu
import matplotlib.pyplot as plt
import os

from shapely.geometry import Polygon, box
from shapely.geometry import MultiPoint
# # TO DO
#
# - validate using reference solution, as did before - DONE
# - compute at each time, the residual of full NS (divergence should be ok)
# - fai interpolazione fine - DONE
# 
#
# # # # 

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

def define_penalty_region(nx, ny, region_type='circle', x_center=0.5, y_center=0.5, radius=0.1):
    """
    Define a penalty region in the domain.
    
    Parameters:
    - nx, ny: grid dimensions (pressure points)
    - region_type: 'circle', 'square', or custom function
    - For circle: x_center, y_center, radius
    - For square: x_center, y_center, width, height
    
    Returns: mask array where 1 indicates penalty region
    """
    mask = np.zeros((nx, ny))
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    xx, yy = np.meshgrid(x, y)
    
    if region_type == 'circle':
        mask = ((xx - x_center)**2 + (yy - y_center)**2 <= radius**2).astype(float)
    elif region_type == 'square':
        width, height = radius, radius  # reuse radius parameter
        mask = ((np.abs(xx - x_center) <= width) & (np.abs(yy - y_center) <= height)).astype(float)
    
    return mask

def compute_cfl_timestep(lx, ly, nx, ny, u_max, v_max, cfl_target=0.3):
    """
    Compute maximum stable time step based on CFL condition.
    
    CFL = max(|u|) * dt / dx < cfl_target
    
    Parameters:
    - lx, ly: domain size
    - nx, ny: grid resolution
    - u_max, v_max: maximum velocities (can be boundary or penalty velocities)
    - cfl_target: target CFL number (typically 0.3-0.5 for stability)
    
    Returns: maximum stable dt
    """
    dx = lx / nx
    dy = ly / ny
    
    vel_max = max(abs(u_max), abs(v_max))
    if vel_max < 1e-10:
        vel_max = 1.0  # default for initial condition
    
    dt_cfl = cfl_target * min(dx, dy) / vel_max
    
    print(f"CFL-limited timestep: dt = {dt_cfl:.6f} (u_max = {u_max}, v_max = {v_max})")
    return dt_cfl

def solve(lx, ly, nx, ny, Nt, vel0, penalty_method=False, penalty_param=None, desired_velocity=(0.0, 0.0), indicator_masks=None):

    dx = lx / nx
    dy = ly / ny

    p = np.zeros([ny + 2, nx + 2])      # N.B. i e j sono invertiti!!!!

    u = np.zeros_like(p)
    ut = np.zeros_like(p) # velocità u tilde
    v = np.zeros_like(p)
    vt = np.zeros_like(p) # velocità v tilde
    divut = np.zeros_like(p)  # divergenza di u tilde, mi serve per il passaggio dove calcolo laplaciano pressione

    Ut, Ub, Vl, Vr = vel0
    
    # Desired velocity in penalty region
    u_desired, v_desired = desired_velocity

    # Penalty method setup
    penalty_mask_u = np.zeros((ny, nx-1))
    penalty_mask_v = np.zeros((ny-1, nx))
    penalty_coeff = 1e4  # penalty coefficient (increase for stronger enforcement)
    
    # Option 1: Use indicator masks from assembly_indicator (for arbitrary polygons)
    if indicator_masks is not None:
        Ix, Iy = indicator_masks
        # Ix has shape (nx+1, ny) - for u on vertical faces
        # Iy has shape (nx, ny+1) - for v on horizontal faces
        # Need to transpose and slice to match interior velocity points
        # u[1:ny+1, 2:nx+1] has shape (ny, nx-1)
        # v[2:ny+1, 1:nx+1] has shape (ny-1, nx)
        penalty_mask_u = Ix[1:-1, :].T  # shape: (ny, nx-1)
        penalty_mask_v = Iy[:, 1:-1].T  # shape: (ny-1, nx)
        penalty_method = True
    # Option 2: Use predefined shapes (circle, square)
    elif penalty_method and penalty_param is not None:
        region_type, region_args = penalty_param
        base_mask = define_penalty_region(nx, ny, region_type, *region_args)
        # Masks on velocity grids (staggered)
        # u[1:ny+1, 2:nx+1] has shape (ny, nx-1)
        # v[2:ny+1, 1:nx+1] has shape (ny-1, nx)
        penalty_mask_u = base_mask[:, :-1]  # u points
        penalty_mask_v = base_mask[:-1, :]  # v points
        if 'penalty_coeff' in region_args:
            penalty_coeff = region_args['penalty_coeff']

    usol=[]
    usol.append(u.copy())

    vsol=[]
    vsol.append(v.copy())

    psol = []
    psol.append(p.copy())

    A_lapl, _ = assemble_lapl(nx, ny, dx, dy)

    # TIME ADVANCE LOOP
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
        
        ut[1:ny+1, 2:nx+1] = u[1:ny+1, 2:nx+1] + dt * (convection + diffusion)
        
        # Penalty method for u (implicit treatment for stability)
        if penalty_method:
            ut[1:ny+1, 2:nx+1] = (ut[1:ny+1, 2:nx+1] + dt * penalty_coeff * penalty_mask_u * u_desired) / (1.0 + dt * penalty_coeff * penalty_mask_u)
           


        ve = 0.5 * (v[2:ny+1, 2:nx+2] + v[2:ny+1, 1:nx+1])
        vw = 0.5 * (v[2:ny+1, 1:nx+1] + v[2:ny+1, 0:nx])
        ue = 0.5 * (u[2:ny+1, 2:nx+2] + u[1:ny, 2:nx+2])
        uw = 0.5 * (u[2:ny+1, 1:nx+1] + u[1:ny, 1:nx+1])
        vn = 0.5 * (v[3:ny+2, 1:nx+1] + v[2:ny+1, 1:nx+1])
        vs = 0.5 * (v[2:ny+1, 1:nx+1] + v[1:ny, 1:nx+1])
        convection = -(ue * ve - uw * vw)/dx - (vn**2 - vs**2)/dy
        diffusion = nu * ((v[2:ny+1, 2:nx+2] - 2.0 * v[2:ny+1, 1:nx+1] + v[2:ny+1, 0:nx]) / dx**2 +
                        (v[3:ny+2, 1:nx+1] - 2.0 * v[2:ny+1, 1:nx+1] + v[1:ny, 1:nx+1]) / dy**2)
        
        vt[2:ny+1, 1: nx+1] = v[2:ny+1, 1: nx+1] + dt*(convection + diffusion)
        
        # Penalty method for v (implicit treatment for stability)
        if penalty_method:
            vt[2:ny+1, 1: nx+1] = (vt[2:ny+1, 1: nx+1] + dt * penalty_coeff * penalty_mask_v * v_desired) / (1.0 + dt * penalty_coeff * penalty_mask_v)
    
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
    
def visual(usol, vsol, psol, lx, nx, dx, ly, ny, dy, penalty_region=None):
    x = np.linspace(dx/2, lx - dx/2, nx)
    y = np.linspace(dy/2, ly - dy/2, ny)
    xx, yy = np.meshgrid(x, y)
    
    u = usol[-1]
    v = vsol[-1]
    speed = np.sqrt(u*u + v*v)
    
    # Figure 1: Streamplot with penalty circle
    fig, ax = plt.subplots(figsize=[6, 6], dpi=150)
    plt.streamplot(xx, yy, u, v, color=speed, density=1.5, cmap=plt.cm.autumn, linewidth=1.0)
    plt.colorbar(label='Speed')
    
    # Draw penalty region (circle)
    if penalty_region is not None:
        region_type, region_args = penalty_region
        if region_type == 'circle':
            x_c, y_c, radius = region_args
            theta = np.linspace(0, 2*np.pi, 100)
            circle_x = x_c + radius * np.cos(theta)
            circle_y = y_c + radius * np.sin(theta)
            plt.plot(circle_x, circle_y, 'b-', linewidth=2, label='Penalty region')
            plt.fill(circle_x, circle_y, alpha=0.2, color='blue')
        elif region_type == 'square':
            x_c, y_c, half_width = region_args
            rect_x = [x_c - half_width, x_c + half_width, x_c + half_width, x_c - half_width, x_c - half_width]
            rect_y = [y_c - half_width, y_c - half_width, y_c + half_width, y_c + half_width, y_c - half_width]
            plt.plot(rect_x, rect_y, 'b-', linewidth=2, label='Penalty region')
            plt.fill(rect_x, rect_y, alpha=0.2, color='blue')
    
    plt.xlim(0, lx)
    plt.ylim(0, ly)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Streamplot with Penalty Region')
    plt.legend()
    plt.gca().set_aspect('equal')
    plt.show()
    
    # Figure 2: Quiver plot with penalty circle
    fig, ax = plt.subplots(figsize=[6, 6], dpi=150)
    skip = 2  # skip every n arrows for clarity
    plt.quiver(xx[::skip, ::skip], yy[::skip, ::skip], u[::skip, ::skip], v[::skip, ::skip], 
               speed[::skip, ::skip], cmap=plt.cm.autumn)
    plt.colorbar(label='Speed')
    
    # Draw penalty region (circle)
    if penalty_region is not None:
        region_type, region_args = penalty_region
        if region_type == 'circle':
            x_c, y_c, radius = region_args
            theta = np.linspace(0, 2*np.pi, 100)
            circle_x = x_c + radius * np.cos(theta)
            circle_y = y_c + radius * np.sin(theta)
            plt.plot(circle_x, circle_y, 'b-', linewidth=2, label='Penalty region')
            plt.fill(circle_x, circle_y, alpha=0.2, color='blue')
        elif region_type == 'square':
            x_c, y_c, half_width = region_args
            rect_x = [x_c - half_width, x_c + half_width, x_c + half_width, x_c - half_width, x_c - half_width]
            rect_y = [y_c - half_width, y_c - half_width, y_c + half_width, y_c + half_width, y_c - half_width]
            plt.plot(rect_x, rect_y, 'b-', linewidth=2, label='Penalty region')
            plt.fill(rect_x, rect_y, alpha=0.2, color='blue')
    
    plt.xlim(0, lx)
    plt.ylim(0, ly)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Quiver Plot with Penalty Region')
    plt.legend()
    plt.gca().set_aspect('equal')
    plt.show()


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

    # Obstacle
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

    Pos_x = P[0, :] + d0x 
    Pos_y = P[1, :] + d0y 


     # Domain
    lx, ly = 1.0, 1.0

    # Number of volumes per direction (pressure volumes)
    nx, ny = 50, 50
    nu = 0.005

    #Initial velocty [Ut, Ub, Vl, Vr]
    vel0 = [1.0, 0.0, 0.0, 0.0]

    dx = lx/nx
    dy = ly/ny

    Ix, Iy =  assembly_indicator(Pos_x, Pos_y, dx, dy, nx_s, ny_s, nx, ny, lx, ly)
    
    # Define desired velocity in penalty region
    u_desired, v_desired = 0.0, 0.0
    
    # Compute CFL-limited time step
    u_max = max(abs(vel0[0]), abs(vel0[1]), abs(u_desired))
    v_max = max(abs(vel0[2]), abs(vel0[3]), abs(v_desired))
    dt = compute_cfl_timestep(lx, ly, nx, ny, u_max, v_max, cfl_target=0.3)
    
    # Total simulation time and number of steps
    T_final = 5.0  # total simulation time
    Nt = int(T_final / dt)
    print(f"Running {Nt} time steps with dt = {dt:.6f}")
    
    # PENALTY METHOD SETUP
    # Option 1: No penalty (standard cavity)
    # usol, vsol, psol = solve(lx, ly, nx, ny, Nt, vel0, penalty_method=False)
    
    # Option 2: Circular penalty region
    # penalty_param = ('circle', [0.7, 0.7, 0.15])  # center_x, center_y, radius
    # usol, vsol, psol = solve(lx, ly, nx, ny, Nt, vel0, penalty_method=True, penalty_param=penalty_param, desired_velocity=(u_desired, v_desired))
    
    # Option 3: Square penalty region
    # penalty_param = ('square', [0.5, 0.5, 0.15])  # center_x, center_y, half_width
    # usol, vsol, psol = solve(lx, ly, nx, ny, Nt, vel0, penalty_method=True, penalty_param=penalty_param, desired_velocity=(u_desired, v_desired))
    
    # Option 4: Use indicator masks from assembly_indicator (arbitrary polygon obstacle)
    usol, vsol, psol = solve(lx, ly, nx, ny, Nt, vel0, desired_velocity=(u_desired, v_desired), indicator_masks=(Ix, Iy))
    
    u_i, v_i, p_i = interpolation(usol, vsol, psol, False)

    # Plot with obstacle mesh
    x = np.linspace(dx/2, lx - dx/2, nx)
    y = np.linspace(dy/2, ly - dy/2, ny)
    xx, yy = np.meshgrid(x, y)
    
    u = u_i[-1]
    v = v_i[-1]
    speed = np.sqrt(u*u + v*v)
    
    fig, ax = plt.subplots(figsize=[6, 6], dpi=150)
    plt.streamplot(xx, yy, u, v, color=speed, density=1.5, cmap=plt.cm.autumn, linewidth=1.0)
    plt.colorbar(label='Speed')
    
    # Draw obstacle mesh
    for e in range(T.shape[1]):
        node_ids = T[:, e]
        x_coords = [Pos_x[i] for i in node_ids] + [Pos_x[node_ids[0]]] 
        y_coords = [Pos_y[i] for i in node_ids] + [Pos_y[node_ids[0]]]
        plt.fill(x_coords, y_coords, alpha=0.5, color='blue')
        plt.plot(x_coords, y_coords, color='darkblue', linewidth=0.5)
    
    plt.xlim(0, lx)
    plt.ylim(0, ly)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Streamplot with Polygon Obstacle (Penalty Method)')
    plt.gca().set_aspect('equal')
    plt.show()
    
    # Quiver plot
    fig, ax = plt.subplots(figsize=[6, 6], dpi=150)
    skip = 2
    plt.quiver(xx[::skip, ::skip], yy[::skip, ::skip], u[::skip, ::skip], v[::skip, ::skip], 
               speed[::skip, ::skip], cmap=plt.cm.autumn)
    plt.colorbar(label='Speed')
    
    # Draw obstacle mesh
    for e in range(T.shape[1]):
        node_ids = T[:, e]
        x_coords = [Pos_x[i] for i in node_ids] + [Pos_x[node_ids[0]]] 
        y_coords = [Pos_y[i] for i in node_ids] + [Pos_y[node_ids[0]]]
        plt.fill(x_coords, y_coords, alpha=0.5, color='blue')
        plt.plot(x_coords, y_coords, color='darkblue', linewidth=0.5)
    
    plt.xlim(0, lx)
    plt.ylim(0, ly)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Quiver Plot with Polygon Obstacle (Penalty Method)')
    plt.gca().set_aspect('equal')
    plt.show()
