from brownian_model import BrownianGlitchModel
import numpy as np
import multiprocessing as mp

# A global variable to hold the model instance in each worker process
system_instance = None

def init_worker(model_params):
    """
    Initializes a model instance once per worker process.
    This avoids the overhead of creating a new model for every single simulation task.
    """
    global system_instance
    system_instance = BrownianGlitchModel(**model_params)
    
def run_single_simulation_worker(sim_args):
    """
    Runs a single simulation using the pre-initialized model instance.
    It only receives arguments that change per simulation (e.g., the seed).
    """
    global system_instance
    x0, T_sim, N_steps, seed = sim_args
    
    try:
        # It's important to re-seed the RNG for each simulation to ensure independence
        system_instance.rng = np.random.Generator(np.random.PCG64(seed))
        # Run the simulation
        return system_instance.simulate(x0, T_sim, N_steps)
    
    except Exception as e:
        print(f"Error in worker process (seed {seed}): {e}")
        return None

def run_single_simulation(args):
    """
    Runs a single BrownianGlitchModel simulation.
    Designed to be used with multiprocessing.Pool.map.
    """
    
    xi, sigma, Xc, dist_type, dist_params, seed, x0, T_sim, N_steps, only_waits = args
    
    try:
        # Create a system instance for each process
        system = BrownianGlitchModel(Xc=Xc, xi=xi, sigma=sigma, dist_type=dist_type, dist_params=dist_params, seed=seed, only_waits=only_waits)
        # Run the simulation
        return system.simulate(x0, T_sim, N_steps)
    
    except Exception as e:
        print(f"Error in worker process (seed {seed}): {e}")
        return None

def simulation_parallel(system_params, x0, Nsim, Tsim=50, Nsteps=500, only_waits=False):
    """
    Runs multiple BrownianGlitchModel simulations in parallel to collect statistics.
    
    Parameters:
    -----------
    system_params : dict
        Dictionary containing model parameters (xi, sigma, Xc, dist_type, dist_params)
    x0 : float
        Initial stress value
    omega : tuple
        Range of stress values to bin (min, max)
    Nbins : int
        Number of bins for histogram
    Nsim : int
        Number of simulations to run
    T_sim : float
        Simulation time length
    Nsteps : int
        Number of time steps
    """

    # Prepare arguments for each simulation run
    base_rng = np.random.default_rng() # RNG for generating seeds
    seeds = base_rng.integers(low=0, high=2**31, size=Nsim)

    # Create list of argument tuples for run_single_simulation
    simulation_args = [(
        system_params['xi'],
        system_params['sigma'],
        system_params['Xc'],
        system_params['dist_type'],
        system_params['dist_params'],
        seed,
        x0,
        Tsim,
        Nsteps,
        only_waits
    ) for seed in seeds]

    # Determine number of processes
    num_processes = max(1, mp.cpu_count() - 1)

    # Use multiprocessing Pool
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(run_single_simulation, simulation_args)

    return results