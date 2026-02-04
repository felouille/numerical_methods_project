import numpy as np
import matplotlib.pyplot as plt


# Simulation parameters
TIME_INTERVAL = 0.001
SPACE_INTERVAL = 0.01
TOTAL_TIME = 1

CELERITY = 1 # Speed of sound in the medium
# FCRIT = CELERITY * TIME_INTERVAL / SPACE_INTERVAL  # Courant-Friedrichs-Lewy condition
# print("FCRIT: ", FCRIT, " which is < 0.8 for stability")

RHO_1 = 1.0  # Density in region 1
RHO_2 = 0.1  # Density in region 2
THRESHOLD = RHO_2 * 1.1
BOLTZ = 1.0  # Boltzmann constant (arbitrary units)
POTENTIAL = 1.0  # Chemical potential (arbitrary units)


def initialize_grid(interval, xmin=0.0, xmax=1.0):
    """Initialize a spacial 1D grid with given interval."""
    return np.arange(xmin, xmax+interval, interval)


def initialize_velocity(grid):
    """Initialize velocity profile on the grid."""
    return np.ones_like(grid)


#Define grid and initial velocity vector
grid = initialize_grid(SPACE_INTERVAL)
VELOCITY = np.zeros_like(grid)

# Define a non-linear velocity profile
#VELOCITY_NONLINEAR = VELOCITY.copy()
#for x in range(len(grid)):
#    VELOCITY_NONLINEAR[x] = (x/len(grid)) # Constant velocity to the right
#VELOCITY = -VELOCITY  # Reverse direction of velocity


#Define initial density profile
#RHO = np.ones_like(grid)  
# Initial condition: Gaussian centered at 0.5
# for x in range(len(grid)):
#     RHO[x] = np.exp(-((grid[x]-0.5)**2)/(2*0.05**2))  

RHO = np.zeros_like(grid)
for x in range(len(grid)):
    if x/len(grid) < 0.5:
        RHO[x] = RHO_1
    else:
        RHO[x] = RHO_2 


# Define initial celerity profile
# CELERITY = np.ones_like(grid)


# Define initial energy profile
# ENERGY = np.ones_like(grid)
# for x in range(len(grid)):
#     ENERGY[x] = (3/2)*RHO[x]*CELERITY[x]**2



def smoothing(vector, smoothing_factor):
    """Apply a simple smoothing filter to the vector."""
    smoothed_vector = vector.copy()
    smoothed_vector[0] = (1-2*smoothing_factor)*vector[0] + 2*smoothing_factor*vector[1]
    smoothed_vector[-1] = (1-2*smoothing_factor)*vector[-1] + 2*smoothing_factor*vector[-2]
    for i in range(1, len(vector)-1):
        smoothed_vector[i] = (1-2*smoothing_factor)*vector[i] + smoothing_factor*(vector[i-1] + vector[i+1])
    return smoothed_vector


#Compute the advection equation
def advection_centered(rho, velocity, time_interval, space_interval, grid, total_time):
    """Compute the advection equation using centered finite difference method."""
    plt.figure(figsize=(10, 6))
    plt.title('Advection Equation Simulation')
    plt.xlabel('Position')
    plt.ylabel('Density')
    rho_next = rho.copy()
    for t in range(0, int(total_time/time_interval)):
        for x in range(len(grid)-1):
            if time_interval*x==0:
                rho_next[x] = rho[x] # Boundary condition at the start
            else:
                rho_next[x] = rho[x] - time_interval*(rho[x+1]*velocity[x+1] - rho[x-1]*velocity[x-1])/(2*space_interval)
            rho_next[-1] = rho[-1]  # Boundary condition at the end
        rho = rho_next.copy()
        if np.mod(t, 100) == 0:
            plt.plot(grid, rho, label=f'Timestep={t}')
            plt.legend()
                      
    return ("Finished")

def advection_upwind(rho, velocity, time_interval, space_interval, grid, total_time):
    """Compute the advection equation using upwind finite difference method."""
    plt.figure(figsize=(10, 6))
    plt.title('Advection Equation Simulation (Upwind)')
    plt.xlabel('Position')
    plt.ylabel('Density')
    rho_next = rho.copy()
    for t in range(0, int(total_time/time_interval)):
        for x in range(len(grid)-1):
            if time_interval*x==0:
                rho_next[x] = rho[x] # Boundary condition at the start
            else:
                rho_next[x] = rho[x] - time_interval*(rho[x]*velocity[x] - rho[x-1]*velocity[x-1])/(space_interval)
            rho_next[-1] = rho[-1]  # Boundary condition at the end
        rho = rho_next.copy()
        if np.mod(t, 100) == 0:
            plt.plot(grid, rho, label=f'Timestep={t}')
            plt.legend()
                     
    return ("Finished")

def advection_downwind(rho, velocity, time_interval, space_interval, grid, total_time):
    """Compute the advection equation using uphill finite difference method."""
    plt.figure(figsize=(10, 6))
    plt.title('Advection Equation Simulation (Uphill)')
    plt.xlabel('Position')
    plt.ylabel('Density')
    rho_next = rho.copy()
    for t in range(0, int(total_time/time_interval)):
        for x in range(len(grid)-1):
            if time_interval*x==0:
                rho_next[x] = rho[x] # Boundary condition at the start
            else:
                rho_next[x] = rho[x] - time_interval*(rho[x+1]*velocity[x+1] - rho[x]*velocity[x])/(space_interval)
            rho_next[-1] = rho[-1]  # Boundary condition at the end
        rho = rho_next.copy()
        if np.mod(t, 100) == 0:
            plt.plot(grid, rho, label=f'Timestep={t}')
            plt.legend()
            plt.show()      
    return ("Finished")



def advect_momentum_centered(rho, velocity, time_interval, space_interval, grid, total_time):
    """Compute the advection equation for momentum."""
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 1.5)

    rho_next = rho.copy()
    velocity_next = velocity.copy()

    for t in range(0, int(total_time/time_interval)):
        rho = smoothing(rho, 0.01)
        velocity = smoothing(velocity, 0.01)
        for x in range(len(grid)-1):
            if time_interval*x==0:
                rho_next[x] = rho[x] # Boundary condition at the start
                velocity_next[x] = velocity[x] # Boundary condition at the start
            
            else:
                rho_next[x] = rho[x] - time_interval*(rho[x+1]*velocity[x+1]**2 - rho[x-1]*velocity[x-1]**2)/(2*space_interval)
                velocity_next[x] = velocity[x]*(rho[x]/rho_next[x]) - (time_interval/rho_next[x])*(rho[x+1]*(velocity[x+1]**2 + CELERITY**2) - rho[x-1]*(velocity[x-1]**2 + CELERITY**2))/(2*space_interval)
            
            rho_next[-1] = rho[-1]  # Boundary condition at the end
            velocity_next[-1] = velocity[-1]  # Boundary condition at the end
        velocity = velocity_next.copy()
        rho = rho_next.copy()

        if np.mod(t, 1000) == 0:
            ax.clear()
            ax.set_xlabel('Position')
            ax.set_ylabel('Density')
            ax.plot(grid, rho, label=f'Timestep={t}')
            ax.legend()
            plt.title('Momentum-Advection Equation Simulation')
            plt.pause(0.01)
            plt.show()
    plt.ioff()         
    return ("Finished")



def advect_momentum_upwind(rho, velocity, time_interval, space_interval, grid, total_time):
    """Compute the advection equation and the momentum equation."""

    t1, t2 = 0, 0

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 1.5)

    rho_next = rho.copy()
    velocity_next = velocity.copy()
    for t in range(0, int(total_time/time_interval)):

        rho = smoothing(rho, 0.1)
        velocity = smoothing(velocity, 0.1)
        for x in range(len(grid)-1):
            if time_interval*x==0:
                rho_next[x] = rho[x] # Boundary condition at the start
                velocity_next[x] = velocity[x] # Boundary condition at the start
            else:
                rho_next[x] = rho[x] - time_interval*(rho[x]*velocity[x]**2 - rho[x-1]*velocity[x-1]**2)/(space_interval)
                velocity_next[x] = velocity[x]*(rho[x]/rho_next[x]) - (time_interval/rho_next[x])*(rho[x]*(velocity[x]**2 + CELERITY**2) - rho[x-1]*(velocity[x-1]**2 + CELERITY**2))/(space_interval)
            rho_next[-1] = rho[-1]  # Boundary condition at the end
            velocity_next[-1] = velocity[-1]  # Boundary condition at the end
        velocity = velocity_next.copy()
        rho = rho_next.copy()

        if rho[int(0.6/SPACE_INTERVAL)] > THRESHOLD and t1 == 0:
            t1 = t
        
        if rho[int(0.7/SPACE_INTERVAL)] > THRESHOLD and t2 == 0:
            t2 = t

        if np.mod(t, 10) == 0:
            ax.clear()
            ax.set_xlim(0, 1)
            ax.set_ylim(0.0, 1.5)
            ax.set_xlabel('Position')
            ax.set_ylabel('Density')
            ax.plot(grid, rho, label=f'Timestep={t}')
            ax.legend()
            plt.title('Momentum-Advection Equation Simulation')
            plt.pause(0.05)
        if t == 200:
            shock_velocity = 0.1/((t2 - t1)*TIME_INTERVAL)
            bedjin = ((shock_velocity / CELERITY)**2) * np.exp((shock_velocity / CELERITY) - (CELERITY / shock_velocity))
            ratio = RHO_1 / RHO_2
            print(bedjin," is approximately equal to the ratio of densities: ", ratio)
            print("Shock velocity: ", shock_velocity)
            return(shock_velocity)
    plt.ioff()
    # plt.show()

    return (shock_velocity)



def advect_momentum_energy_centered(rho, velocity, energy, celerity, time_interval, space_interval, grid, total_time):
    """Compute the advection equation for momentum and energy."""
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))

    rho_next = rho.copy()
    velocity_next = velocity.copy()
    energy_next = energy.copy()
    celerity_next = celerity.copy()

    for t in range(0, int(total_time/time_interval)):

        rho = smoothing(rho, 0.002)
        velocity = smoothing(velocity, 0.002)
        energy = smoothing(energy, 0.002)
        celerity = smoothing(celerity, 0.002)

        for x in range(len(grid)-1):
            if time_interval*x==0:
                rho_next[x] = rho[x] # Boundary condition at the start
                velocity_next[x] = velocity[x] # Boundary condition at the start
                energy_next[x] = energy[x] # Boundary condition at the start
                celerity_next[x] = celerity[x] # Boundary condition at the start

            else:
                rho_next[x] = rho[x] - time_interval*(rho[x+1]*velocity[x+1]**2 - rho[x-1]*velocity[x-1]**2)/(2*space_interval)
                velocity_next[x] = velocity[x]*(rho[x]/rho_next[x]) - (time_interval/rho_next[x])*(rho[x+1]*(velocity[x+1]**2 + celerity[x+1]**2) - rho[x-1]*(velocity[x-1]**2 + celerity[x-1]**2))/(2*space_interval)
                energy_next[x] = energy[x] - time_interval*((energy[x+1] + rho[x]*celerity[x]**2)*velocity[x+1] - (energy[x-1] + rho[x-1]*celerity[x-1]**2)*velocity[x-1])/(2*space_interval)
                celerity_next[x] = np.sqrt((1/3)* ((2*energy_next[x]/rho_next[x]) - velocity_next[x]**2))

            rho_next[-1] = rho[-1]  # Boundary condition at the end
            velocity_next[-1] = velocity[-1]  # Boundary condition at the end
            energy_next[-1] = energy[-1]  # Boundary condition at the end
            celerity_next[-1] = celerity[-1]  # Boundary condition at the end

        velocity = velocity_next.copy()
        rho = rho_next.copy()
        energy = energy_next.copy()
        celerity = celerity_next.copy()

        if np.mod(t, 100) == 0:
            ax.clear()
            ax.set_xlim(0, 1)
            ax.set_ylim(0.0, 3.0)
            ax.set_xlabel('Position')
            ax.set_ylabel('Density')
            ax.plot(grid, celerity, label=f'Timestep={t}')
            ax.legend()
            plt.title('Momentum-Advection Equation Simulation')
            plt.pause(0.01)

    plt.ioff()
    plt.show()         
    return ("Finished")
    


def advect_nebulae(rho, velocity, celerity, time_interval, space_interval, grid, total_time):
    """Application to the expansion of a planetary nebulae"""
    fig, ax = plt.subplots(nrows=2,ncols=1, figsize=(10, 6))

    rho_next = rho.copy()
    velocity_next = velocity.copy()

    for t in range(0, int(total_time/time_interval)):

        rho = smoothing(rho, 0.01)
        velocity = smoothing(velocity, 0.01)

        for x in range(len(grid)-1):
            if (x==0):
                rho_next[x] = rho[x] # Boundary condition at the start
                velocity_next[x] = velocity[x] # Boundary condition at the start

            else:
                rho_next[x] = rho[x] - time_interval*((rho[x+1]*velocity[x+1] - rho[x-1]*velocity[x-1])/(2*space_interval) + (2*rho[x]*velocity[x])/(x*space_interval))
                velocity_next[x] = velocity[x]*(rho[x]/rho_next[x]) - (time_interval/rho_next[x])*((rho[x+1]*(velocity[x+1]**2 + celerity**2) - rho[x-1]*(velocity[x-1]**2 + celerity**2))/(2*space_interval) + (2*rho[x]*(velocity[x]**2))/(x*space_interval))
            
            rho_next[-1] = rho[-1]  # Boundary condition at the end
            velocity_next[-1] = velocity[-1]  # Boundary condition at the end

        velocity = velocity_next.copy()
        rho = rho_next.copy()

        if (np.mod(t, 500) == 0) & (t < 5000):
            ax[0].set_title('Density Profile')
            ax[0].set_xlabel('Position (m)')
            ax[0].set_ylabel('Hydrogen atom density (cm$^{-3}$)')
            ax[0].grid()
            ax[0].set_yscale('log')
            ax[0].set_xlim(0, 0.1*3e16)
            ax[0].plot(grid, [x/1.66e-21 for x in rho], label=f'Timestep={int(t*5e7/31557600)} (years)')
            ax[0].legend()

            ax[1].set_title('Velocity Profile')
            ax[1].set_xlabel('Position (m)')
            ax[1].set_ylabel('Velocity (m/s)')
            ax[1].set_xlim(0, 0.1*3e16)
            ax[1].set_ylim(0, 60000)
            ax[1].plot(grid, velocity, label=f'Timestep={int(t*5e7/31557600)} (years)')
            ax[1].legend()
    
        elif t > 5000:
            ax[1].plot(grid, [80000/11e15 * x for x in grid], label='Data for 8000 years', color='green', linestyle='--')
            ax[1].plot(grid, [90000/36e15 * x for x in grid], label='Data for 13000 years', color='red', linestyle='--')
            ax[1].legend()
            return("Finished")  
    return ("Finished")
### References to compare results:
### DOI : 10.1086/148488
### DOI : 10.1086/149557




def is_conserving_mass(rho, velocity, time_interval, space_interval, grid, total_time):
    """Check if the advection equation conserves mass."""
    mass = []
    mass.append(np.sum(rho) * space_interval)
    rho_next = rho.copy()
    for t in range(0, int(total_time/(time_interval))):
        for x in range(1, len(grid)-1):
            rho_next[x] = rho[x] - time_interval*(rho[x+1]*velocity[x+1] - rho[x-1]*velocity[x-1])/(2*space_interval)
        rho_next[0] = rho[0]  # Boundary condition at the start
        rho_next[-1] = rho[-1]  # Boundary condition at the end
        rho = rho_next.copy()
        mass.append(np.sum(rho) * space_interval)
    final_mass = mass[-1]
    print("Initial Mass: ", mass[0])
    print("Final Mass: ", final_mass)
    time_steps = np.arange(0, total_time + time_interval, time_interval)
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, mass)
    plt.title('Mass Conservation Check')
    plt.xlabel('Time')
    plt.ylabel('Total Mass')
    plt.show()
    return(3)



def conservative_formulation(rho, velocity, celerity, time_interval, space_interval, grid, total_time):
    """Compute the momentum and advection equations while conserving mass."""
    
    fig, ax = plt.subplots(nrows=2,ncols=2, figsize=(10, 6))
    fig.suptitle('Conservative Formulation Simulation (isothermal case)')
    smoothing_factor = 0

    ### Vectors initialization
    celerity_next = celerity.copy()
    mass = [density*space_interval**3 for density in rho]
    mass_next = mass.copy()
    Mv = np.zeros_like(velocity)
    Mv_next = Mv.copy()
    total_mass = []
    time_steps = []
    ### Velocity borders initialization
    velocities_borders = np.ones_like(velocity)
    for k in range(1, len(velocity)):
        velocities_borders[k] = (velocity[k] + velocity[k-1]) / 2
    velocities_borders[0] = velocity[0]
    velocities_borders[-1] = velocity[-1]

    ### Main loop
    for t in range(0, int(total_time/time_interval)):
        mass = smoothing(mass, smoothing_factor)
        Mv = smoothing(Mv, smoothing_factor)

        for x in range(len(grid)):
            
            ### Boundary conditions
            if x==0 or x==len(grid)-1:
                Mv_next[x] = 0
            
            if x==0:
                mv_left = 0
                mv_right = rho[x] * velocities_borders[x+1] * space_interval**2 if velocities_borders[x] > 0 else rho[x+1] * velocities_borders[x+1] * space_interval**2
                mass_next[x] = mass[x] + time_interval * (mv_left - mv_right)
            
            if x==len(grid)-1:
                mv_left = rho[x-1] * velocities_borders[x] * space_interval**2 if velocities_borders[x] > 0 else rho[x] * velocities_borders[x]  * space_interval**2
                mv_right = 0
                mass_next[x] = mass[x] + time_interval * (mv_left - mv_right)

            else:
                ### Equation for the change of mass
                mv_left = rho[x-1] * velocities_borders[x] * space_interval**2 if velocities_borders[x] > 0 else rho[x] * velocities_borders[x]  * space_interval**2
                mv_right = rho[x] * velocities_borders[x+1] * space_interval**2 if velocities_borders[x] > 0 else rho[x+1] * velocities_borders[x+1] * space_interval**2
                mass_next[x] = mass[x] + time_interval * (mv_left - mv_right)
                
                ### Equation for the change of momentum
                mvv_left = rho[x-1] * velocity[x-1] * velocities_borders[x] * space_interval**2 if velocities_borders[x] > 0 else rho[x] * velocity[x] * velocities_borders[x] * space_interval**2
                mvv_right = rho[x] * velocity[x] * velocities_borders[x+1] * space_interval**2 if velocities_borders[x] > 0 else rho[x+1] * velocity[x+1] * velocities_borders[x+1] * space_interval**2
    
                pressure_left = rho[x-1] * (celerity[x-1]**2) * space_interval**2    #In reality we compute P*A, but I wrote pressure for clarity
                pressure_right = rho[x+1] * (celerity[x+1]**2) * space_interval**2
                Mv_next[x] = Mv[x] + time_interval * (mvv_left - mvv_right + (-pressure_right + pressure_left))

                
        ### Update velocity and density
        for x in range(0, len(grid)):
            velocity[x] = Mv_next[x] / (mass_next[x] + 1e-12)
        velocity[0] = 0
        velocity[-1] = 0
            #print(Mv_next[x], mass_next[x]) if mass_next[x]<1e-6 else None
        
        for x in range(0, len(grid)):
            rho[x] = mass_next[x] / (space_interval**3)
        
        for x in range(1,len(velocity)):
            velocities_borders[x] = (velocity[x-1] + velocity[x]) / 2
        velocities_borders[0] = velocity[0]
        #velocities_borders[-1] = velocity[-1]

        ### Update mass and momentum
        mass = mass_next.copy()
        Mv = Mv_next.copy()

        if (np.mod(t, 40) == 0):
            ax[0][0].set_title('Density Profile')
            ax[0][0].set_xlabel('Position (arbitrary units)')
            ax[0][0].set_ylabel('$\\rho$')
            ax[0][0].grid()
            ax[0][0].plot(grid, rho, label=f'Timestep={t}')
            ax[0][0].legend()

            ax[0][1].set_title('Velocity Profile')
            ax[0][1].set_xlabel('Position (arbitrary units)')
            ax[0][1].set_ylabel('Velocity (arbitrary units)')
            ax[0][1].plot(grid, velocity, label=f'Timestep={t}')
            ax[0][1].legend()
            print(f'Plotting at timestep: {t}')
            
            m = 0
            for density in rho:
                m += density * space_interval**3
            total_mass.append(m)
            time_steps.append(t)

        elif t > 200:
            ax[1][0].set_title('Total Mass Over Time')
            ax[1][0].set_xlabel('Timestep')
            ax[1][0].set_ylabel('Total Mass')
            ax[1][0].plot(time_steps, total_mass, label='Total Mass')
            ax[1][0].legend()
            return(total_mass, time_steps)
    return ("Finished")     


            

            
            


#Main


# THIS PART IS FOR TESTING THE SHOCK VELOCITY AS A FUNCTION OF THE DENSITY RATIO, AND COMPARING IT TO THE THEORETICAL PREDICTION GIVEN BY BEDJIN'S FORMULA
# shock_velocities = []
# RHO_1 = 1.0
# for i in range(1, 8, 1):
#     RHO_2 = i/10
#     RHO = np.zeros_like(grid)
#     THRESHOLD = RHO_2 * 1.05
#     for x in range(len(grid)):
#         if x/len(grid) < 0.5:
#             RHO[x] = RHO_1
#         else:
#             RHO[x] = RHO_2 
#     print("Running simulation for RHO_1 = ", RHO_1, " and RHO_2 = ", RHO_2)
#     shock_velocity = advect_momentum_upwind(RHO, VELOCITY, TIME_INTERVAL, SPACE_INTERVAL, grid, TOTAL_TIME)
#     shock_velocities.append((shock_velocity))

# ratios = np.linspace(1, 10, 50)

# def theoretical_bedjin(vel, ratio):
#     return (vel**2) * np.exp(vel - 1/vel) - ratio

# def theoretical_bedjin_prime(vel):
#     return (2*vel + vel**2 + 1) * np.exp(vel - 1/vel)

# th_velocities = []

# for x in ratios:
#     vel = 10.0
#     for k in range(30):
#         vel = vel - (theoretical_bedjin(vel, x) / theoretical_bedjin_prime(vel))
#     th_velocities.append(vel)


# th_results = []
# for v in th_velocities:
#     th_results.append((v**2) * np.exp(v - 1/v))

# # Plot shock velocities vs density ratios
# density_ratios = [RHO_1/i for i in np.arange(0.1, 0.8, 0.1)]
# plt.figure(figsize=(10, 6))
# plt.plot(density_ratios, shock_velocities, marker='o', label='Simulation Results')
# plt.plot(ratios, th_velocities, label='Theoretical Prediction', linestyle='--')
# plt.xlabel('Density Ratio (RHO_1 / RHO_2)')
# plt.ylabel('Vs / C')
# plt.title('Shock Velocity vs Density Ratio, smoothing factor=0.075')
# #plt.yscale('log')
# plt.legend()
# plt.show()


#grid = initialize_grid(SPACE_INTERVAL, xmin=0.005*3e16, xmax=0.1*3e16) # Grid in meters


VELOCITY=np.zeros_like(grid)


# FOR PLANETARY NEBULAE SIMULATION
# RHO_1 = 2.19*10**(-16) # Density in region 1
# RHO_2 = 1.66*10**(-21) # Density in region 2

# FOR OTHER SIMULATIONS
RHO_1 = 1.0 # Density in region 1
RHO_2 = 0.1 # Density in region 2
RHO = np.zeros_like(grid)
for x in range(len(grid)):
    if x/len(grid) < 0.5:
        RHO[x] = RHO_1
    else:
        RHO[x] = RHO_2



CELERITY = np.ones_like(grid) - 0.5 # Conversion in m/s


# is_conserving_mass(RHO, VELOCITY, TIME_INTERVAL, SPACE_INTERVAL, grid, TOTAL_TIME)
# total_mass, time_steps = conservative_formulation(RHO, VELOCITY, CELERITY, TIME_INTERVAL, SPACE_INTERVAL, grid, TOTAL_TIME)
# plt.show()


