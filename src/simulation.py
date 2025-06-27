import numpy as np
import multiprocessing as mp

# A global variable to hold the model instance in each worker process
system_instance = None

def init_worker(model_class, model_params):
    """
    Initializes a model instance once per worker process.
    This avoids the overhead of creating a new model for every single simulation task.
    """
    global system_instance

    model_init_params = model_params.copy()
    model_init_params['only_waits'] = True
        
    system_instance = model_class(**model_init_params)
    
def run_single_simulation_worker(sim_args):
    """
    Runs a single simulation using the pre-initialized model instance.
    It only receives arguments that change per simulation (e.g., the seed).
    """
    global system_instance
    
    # Unpack arguments
    x0, T_sim, N_steps, seed = sim_args
    
    try:
        # It's important to re-seed the RNG for each simulation to ensure independence
        system_instance.rng = np.random.Generator(np.random.PCG64(seed))
        
        return system_instance.simulate(x0, T_sim, N_steps)
    
    except Exception as e:
        print(f"Error in worker process (seed {seed}): {e}")
        return None

def run_single_simulation(args):
    """
    Runs a single model simulation.
    Designed to be used with multiprocessing.Pool.map.
    """
    model_class, model_params, x0, T_sim, N_steps, seed = args
    
    try:
        # Create a system instance for each process
        model_params_with_seed = {**model_params, 'seed': seed}
        system = model_class(**model_params_with_seed)
        
        return system.simulate(x0, T_sim, N_steps)
    
    except Exception as e:
        print(f"Error in worker process (seed {seed}): {e}")
        return None

def simulation_parallel(model_class, model_params, x0, Nsim, Tsim=50, Nsteps=None):
    """
    Runs multiple model simulations in parallel to collect statistics.
    
    Parameters:
    -----------
    model_class : class
        The model class to use (e.g., BrownianGlitchModel, SDPModel, HybridModel)
    model_params : dict
        Dictionary containing model parameters
    x0 : float
        Initial stress value
    Nsim : int
        Number of simulations to run
    Tsim : float
        Simulation time length
    Nsteps : int, optional
        Number of time steps (required for some models)
    """

    # Prepare arguments for each simulation run
    base_rng = np.random.default_rng() # RNG for generating seeds
    seeds = base_rng.integers(low=0, high=2**31, size=Nsim)

    # Create list of argument tuples for run_single_simulation
    simulation_args = [(
        model_class,
        model_params,
        x0,
        Tsim,
        Nsteps,
        seed
    ) for seed in seeds]

    # Determine number of processes
    num_processes = max(1, mp.cpu_count() - 1)

    # Use multiprocessing Pool
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(run_single_simulation, simulation_args)

    return results