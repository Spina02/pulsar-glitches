import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import os
import pandas as pd
import inspect

from brownian_model import BrownianGlitchModel
from sdp_model import SDPModel
from hybrid_model import HybridModel
from model import Model
from simulation import init_worker, run_single_simulation_worker, simulation_parallel

# Helper function to create consistent filenames
def _create_filename(prefix, params):
    """Creates a standardized filename from a dictionary of parameters."""
    # Flatten nested dictionaries like dist_params
    flat_params = {}
    for key, value in params.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat_params[f"{key}_{sub_key}"] = sub_value
        else:
            flat_params[key] = value

    # Create a string from the parameters
    param_str = "_".join(f"{k}{v}" for k, v in sorted(flat_params.items()))
    # Sanitize the string to be a valid filename
    param_str = param_str.replace('.', 'p').replace('-', 'm')
    return f"{prefix}_{param_str}.csv"


def plot_single_trajectory(model_class, model_params, x0, T, N=None, 
                          title=None, seed=None):
    """
    Plots a single trajectory for any model type.
    
    Parameters:
    -----------
    model_class : class
        The model class to use (e.g., BrownianGlitchModel, SDPModel, HybridModel)
    model_params : dict
        Parameters to initialize the model
    x0 : float
        Initial stress value
    T : float
        Total simulation time
    N : int, optional
        Number of time steps (required for BrownianGlitchModel and HybridModel)
    title : str, optional
        Plot title
    seed : int, optional
        Random seed
    """
    # Create the system and run simulation
    model, result = create_and_simulate_model(model_class, model_params, x0, T, N, seed)
    
    # Plot the trajectory
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(result['times'], result['traj'], color='slateblue', lw=1)
    
    # Plot threshold line if available
    if hasattr(model, 'Xc'):
        ax.axhline(model.Xc, color='red', linestyle='--', linewidth=1)
    
    # Plot glitches as vertical lines at the bottom
    glitch_times = result['glitch_times']
    for glitch_time in glitch_times:
        ax.axvline(glitch_time, ymin=0, ymax=0.05, color='black', linewidth=1)
    
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, T)
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'{model_class.__name__} trajectory')
    ax.set_xlabel(r'Time, $t$ (arb. units)')
    ax.set_ylabel(r'Stress, $X$')
    plt.show()


def plot_glitch_panels(model_class, model_params_list, x0, T, N=None,
                      figsize=(8,6), main_title="Process trajectory", 
                      panel_titles=None, seed=None, n_cols=2):
    """
    Plots multiple trajectory panels for any model type.
    
    Parameters:
    -----------
    model_class : class
        The model class to use (e.g., BrownianGlitchModel, SDPModel, HybridModel)
    model_params_list : list of dict
        List of parameter dictionaries, one for each panel
    x0 : float
        Initial stress value
    T : float
        Total simulation time
    N : int, optional
        Number of time steps (required for BrownianGlitchModel and HybridModel)
    figsize : tuple
        Figure size
    main_title : str
        Main title for the figure
    panel_titles : list of str, optional
        Titles for each panel
    seed : int, optional
        Random seed
    """
    n_panels = len(model_params_list)
    n_rows = (n_panels + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True)
    if n_panels <= n_cols:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx, model_params in enumerate(model_params_list):
        if idx >= n_panels:
            break
            
        # Create the system and run simulation
        model, result = create_and_simulate_model(model_class, model_params, x0, T, N, seed)
        
        ax = axes[idx]
        # Plot trajectory
        ax.plot(result['times'], result['traj'], color='slateblue', lw=1)
        
        # Plot Xc as dashed red line if available
        if hasattr(model, 'Xc'):
            ax.axhline(model.Xc, color='red', linestyle='--', linewidth=1)
        
        # Plot glitches as vertical segments at the bottom
        y0 = 0.05  # bottom of plot for glitch bars
        scale = 0.6
        
        # Normalize glitch sizes for bar heights
        for t, s in zip(result['glitch_times'], result['glitch_sizes']):
            bar_height = y0 + s * scale
            ax.vlines(t, 0, bar_height, color='k', linewidth=1)
            
        # Set limits and labels
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, T)
        
        # Set panel title
        if panel_titles and idx < len(panel_titles):
            ax.set_title(panel_titles[idx], fontsize=16)
        else:
            ax.set_title(f'Panel {idx+1}', fontsize=16)
            
        if idx % n_cols == 0:
            ax.set_ylabel(r'Stress, $X$')
        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel(r'Time, $t$ (arb. units)')
    
    # Hide unused axes
    for idx in range(n_panels, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(main_title, fontsize=25)
    plt.tight_layout()
    plt.show()


def plot_waiting_time_histogram(model_class, model_params_list, x0, T, N=None,
                               Nsim=100, panel_titles=None, seed=None,
                               figsize=(12, 10), xlim=(0, 25), ylim=(0, 0.7)):
    """
    Plots waiting time histograms for multiple model configurations.
    
    Parameters:
    -----------
    model_class : class
        The model class to use
    model_params_list : list of dict
        List of parameter dictionaries, one for each panel
    x0 : float
        Initial stress value
    T : float
        Total simulation time
    N : int, optional
        Number of time steps (required for some models)
    Nsim : int
        Number of simulations per panel
    panel_titles : list of str, optional
        Titles for each panel
    seed : int, optional
        Random seed
    figsize : tuple
        Figure size
    xlim : tuple
        X-axis limits
    ylim : tuple
        Y-axis limits
    """
    n_panels = len(model_params_list)
    n_cols = 2
    n_rows = (n_panels + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, sharey=True)
    if n_panels <= n_cols:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx, model_params in enumerate(model_params_list):
        if idx >= n_panels:
            break
        
        # Collect waiting times from multiple simulations
        all_waiting_times = []
        
        results = simulation_parallel(model_class, model_params, x0, Nsim, 
                                    Tsim=T, Nsteps=N)
            
        # Collect waiting times from all simulation results
        for res in results:
            if res and 'waiting_times' in res and len(res['waiting_times']) > 0:
                all_waiting_times.extend(res['waiting_times'])
        
        all_waiting_times = np.array(all_waiting_times)
        
        ax = axes[idx]
        if len(all_waiting_times) > 0:
            ax.hist(all_waiting_times, bins=100, density=True, alpha=0.5)
        
        ax.set_xlabel(r'Waiting time, $\Delta t$')
        ax.set_ylabel(r'PDF')
        
        # Set panel title
        if panel_titles and idx < len(panel_titles):
            ax.set_title(panel_titles[idx])
        else:
            ax.set_title(f'Panel {idx+1}')
            
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    
    # Hide unused axes
    for idx in range(n_panels, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def plot_waiting_time_distributions(model_class, model_params_base, 
                                  param_variations=None,
                                  colormap=['tab:red', 'tab:orange', 'tab:green', 'tab:blue'],
                                  bins_lin=np.linspace(0.0, 5.0, 40),
                                  bins_log=np.logspace(-2, 2, 40),
                                  x0=0.5, T=1000, N=None,
                                  N_glitches=100_000,
                                  seed=None,
                                  store_results=False,
                                  save_fig=True,
                                  results_dir="results",
                                  labels=None,
                                  load=False,
                                  filepath=None
                                  ):
    """
    Plots waiting time distributions for any model type with different parameter variations.
    
    Parameters:
    -----------
    model_class : class
        The model class to use
    model_params_base : dict
        Base parameters for the model (common to all variations)
    param_variations : list of dict
        List of parameter updates for each variation. Each dict will be merged with base params.
    colormap : list
        List of colors for each variation
    bins_lin, bins_log : array
        Bins for linear and log histograms
    x0 : float
        Initial stress value
    T : float
        Simulation time
    N : int, optional
        Number of time steps (required for some models)
    N_glitches : int
        Number of glitches to collect
    seed : int, optional
        Random seed
    store_results : bool
        Whether to cache results
    save_fig : bool
        Whether to save the figure
    results_dir : str
        Directory for results/figures
    labels : list of str, optional
        Labels for each variation
    """
    # Create the results directory if it doesn't exist
    if store_results or save_fig:
        os.makedirs(results_dir, exist_ok=True)
        
    if len(colormap) != len(param_variations):
        # default colormap, generate a list of colors
        colormap = plt.cm.tab10(np.linspace(0, 1, len(param_variations)))
        
    # If N is an int, make it an array of size #variations with that value; else, make it an array of size #variations
    n_variations = len(param_variations) if param_variations is not None else 1
    if isinstance(N, int):
        N = [N] * n_variations
    else:
        if len(N) != n_variations:
            raise ValueError(f"N must be an int or an array of size {n_variations}")
        
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    
    base_rng = np.random.default_rng(seed)
    
    if param_variations is None:
        param_variations = [{}]  # Single run with base parameters
    
    for idx, (param_update, c) in enumerate(zip(param_variations, colormap)):
        
        # Merge base parameters with variation
        current_model_params = {**model_params_base, **param_update}
        
        if store_results or save_fig or (load and filepath is None):
            # caching results
            params_for_filename = {
                'model': model_class.__name__,
                **current_model_params, 
                'x0': x0, 
                'T': T,
                'N_glitches': N_glitches
            }
            
            filename = _create_filename("waiting_times", params_for_filename)
            filepath = os.path.join(results_dir, filename)

        if load and os.path.exists(filepath):
            print(f"Loading cached results for variation {idx} from {filepath}")
            df = pd.read_csv(filepath)
            delta_t_tot = df['waiting_time'].values
        else:
            dt_list = []
            collected_glitches = 0
            
            max_sims = N_glitches // 10 if N_glitches > 100 else 100
            seeds = base_rng.integers(low=0, high=2**31, size=max_sims)
            
            def arg_generator():
                for s in seeds:
                    yield (x0, T, N[idx], s)

            model_init_args = current_model_params.copy()

            num_processes = max(1, mp.cpu_count() - 1)
            with mp.Pool(processes=num_processes, initializer=init_worker, 
                        initargs=(model_class, model_init_args)) as pool:
                results_iterator = pool.imap_unordered(run_single_simulation_worker, arg_generator())
                for i, result in enumerate(results_iterator):
                    # Handle both direct waiting times and full results
                    waiting_times = result
                        
                    if waiting_times is not None and len(waiting_times) > 0:
                        dt_list.append(waiting_times)
                        collected_glitches += len(waiting_times)
                    print(f"\r[variation {param_update}] Collected {collected_glitches}/{N_glitches} glitches from {i+1} simulations...", end="")
                    if collected_glitches >= N_glitches:
                        pool.terminate()
                        break
            print("\n")
            
            delta_t_tot = np.concatenate(dt_list)[:N_glitches]
            
            if store_results:
                print(f"Storing results to {filepath}")
                pd.DataFrame({'waiting_time': delta_t_tot}).to_csv(filepath, index=False)
        
        # plotting
        if delta_t_tot.mean() > 0:
            delta_t_norm = delta_t_tot / delta_t_tot.mean()
        else:
            delta_t_norm = delta_t_tot

        # Determine label
        if labels and idx < len(labels):
            label = labels[idx]
        else:
            label = f'Variation {idx+1}'

        pdf_lin, edges = np.histogram(delta_t_norm, bins=bins_lin, density=True)
        centers_lin = 0.5 * (edges[1:] + edges[:-1])
        ax[0].plot(centers_lin, pdf_lin, lw=1.4, color=c, label=label)

        pdf_log, edges = np.histogram(delta_t_norm, bins=bins_log, density=True)
        centers_log = np.sqrt(edges[:-1] * edges[1:])
        ax[1].plot(centers_log, pdf_log, lw=1.4, color=c)

    # Formatting axes
    ax[0].set_yscale('log')
    ax[0].set_xlabel(r'$\Delta t / \langle \Delta t \rangle$')
    ax[0].set_ylabel(r'$p(\Delta t)$')
    ax[0].legend(frameon=False, fontsize=9)
    ax[0].set_xlim(0, 5)
    ax[0].set_ylim(1e-2, 1e1)
    ax[0].set_title('log-linear')

    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xlabel(r'$\Delta t / \langle \Delta t \rangle$')
    ax[1].set_xlim(1e-2, 1e2)
    ax[1].set_ylim(1e-5, 1e2)
    ax[1].set_title('log-log')

    plt.tight_layout()
    
    if save_fig:
        dist_type = model_params_base.get('dist_type', 'unknown')
        fig_filename_base = f"wtd_plot_{model_class.__name__}_{dist_type}.png"
        fig_filepath = os.path.join(results_dir, fig_filename_base)
        print(f"Saving figure to {fig_filepath}")
        plt.savefig(fig_filepath, dpi=300)
    
    plt.show()


def plot_model_comparison(models_config, x0=0.1, T=2.5, N_default=5000, 
                         seed=None, figsize=(10, 5)):
    """
    Plots comparison of different models side by side.
    
    Parameters:
    -----------
    models_config : list of dict
        Each dict should contain:
        - 'model_class': The model class
        - 'model_params': Parameters for the model
        - 'title': Title for the panel
        - 'N': Number of steps (optional, uses N_default if not specified)
    x0 : float
        Initial stress value
    T : float
        Simulation time
    N_default : int
        Default number of steps for models that need it
    seed : int, optional
        Random seed
    figsize : tuple
        Figure size
    """
    n_models = len(models_config)
    fig, axes = plt.subplots(2, n_models, figsize=figsize, sharex='col', 
                            gridspec_kw={'height_ratios': [1, 0.6]})
    if n_models == 1:
        axes = axes.reshape(-1, 1)
    
    # Visualization parameters
    v_dot = 0.1  # Arbitrary spin-down rate for visualization
    time_continuous = np.linspace(0, T, 1000)
    baseline_velocity = -v_dot * time_continuous + 0.5

    # Set font sizes
    label_fontsize = 16
    title_fontsize = 24
    tick_fontsize = 14

    for col_idx, config in enumerate(models_config):
        model_class = config['model_class']
        model_params = config['model_params']
        title = config.get('title', model_class.__name__)
        N = config.get('N', N_default)
        
        # Create model and run simulation
        model, result = create_and_simulate_model(model_class, model_params, x0, T, N, seed)
        
        ax_stress = axes[0, col_idx]
        ax_vel = axes[1, col_idx]
        
        # Top panel: Stress Trajectory
        ax_stress.plot(result['times'], result['traj'], lw=1, color='blue')
        if hasattr(model, 'Xc'):
            ax_stress.axhline(model.Xc, color='red', linestyle='--', lw=1, alpha=0.7)
        ax_stress.set_ylim(0, 1.1)
        ax_stress.set_xlim(0, T)
        ax_stress.set_title(title, fontsize=title_fontsize, pad=10)
        
        # Add glitch markers
        for t in result['glitch_times']:
            ax_stress.axvline(t, ymin=0, ymax=0.05, color='k', lw=1)
        
        # Add cumulative glitch effects
        total_velocity = baseline_velocity.copy()
        for t_glitch, glitch_size in zip(result['glitch_times'], result['glitch_sizes']):
            mask = time_continuous >= t_glitch
            total_velocity[mask] += glitch_size * 2e-2
        ax_vel.plot(time_continuous, total_velocity, color='darkorange', lw=2)

        ax_vel.set_xlabel('Time, t (arb. units)', fontsize=label_fontsize)
        ax_vel.set_xlim(0, T)
        ax_vel.set_ylim(0.2, 0.6)
        
        # Set labels for left column only
        if col_idx == 0:
            ax_stress.set_ylabel('Stress, X', fontsize=label_fontsize)
            ax_vel.set_ylabel('Angular velocity\n(arb. units)', fontsize=label_fontsize)

        # Set tick label font sizes
        ax_stress.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        ax_vel.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    fig.tight_layout()
    plt.show()


def plot_sdp_vs_brownian(T=2.5, N=5000, seed=None):
    """Comparison of SDP and Brownian models using the generic framework."""
    
    # Common parameters for both models
    common_params = {
        'Xc': 1.0,
        'dist_type': 'neg_powerlaw', 
        'dist_params': {'delta': 1.5, 'beta': 1e-2}
    }
    
    # Configuration for each model
    models_config = [
        {
            'model_class': SDPModel,
            'model_params': {
                'alpha': 1.0, 
                'f': 1.0,
                **common_params
            },
            'title': 'SDP Model',
            'N': None  # SDP doesn't need N
        },
        {
            'model_class': BrownianGlitchModel,
            'model_params': {
                'sigma': 0.2,
                'xi': 0.2**2 * 50.0,  # mu = 50.0
                **common_params
            },
            'title': 'Brownian Model',
            'N': N
        }
    ]
    
    plot_model_comparison(models_config, x0=0.1, T=T, N_default=N, seed=seed)


# Helper function to handle model instantiation and simulation
def create_and_simulate_model(model_class, model_params, x0, T, N=None, seed=None):
    """
    Creates a model instance and runs simulation with appropriate parameters.
    Handles different model types that may or may not require N parameter.
    """
    # Create model instance
    if seed is not None:
        model_params_with_seed = {**model_params, 'seed': seed}
    else:
        model_params_with_seed = model_params.copy()
    
    model = model_class(**model_params_with_seed)
    
    result = model.simulate(x0, T, N)
    
    return model, result


# Keep the old function for backward compatibility
def plot_single_trajectory_brownian(mu, xi, Xc, dist_type, dist_params, x0, T, N, seed=None):
    """Deprecated: Use plot_single_trajectory with BrownianGlitchModel instead."""
    model_params = {
        'xi': xi,
        'sigma': np.sqrt(xi/mu),
        'Xc': Xc,
        'dist_type': dist_type,
        'dist_params': dist_params
    }
    plot_single_trajectory(BrownianGlitchModel, model_params, x0, T, N, 
                          title=r'$\xi/\sigma^2 = %.1f$' % mu, seed=seed)

# Keep the old function for backward compatibility
def plot_waiting_time_histogram_brownian(mu_list, Xc=1.0, sigma=0.15,
                                dist_type='neg_powerlaw', 
                                dist_params={'delta': 1.5, 'beta': 0.01},
                                x0=0.5, T=500, N=500_000,
                                Nsim=100, seed=None):
    """Deprecated: Use plot_waiting_time_histogram instead."""
    model_params_list = []
    panel_titles = []
    
    for mu in mu_list:
        xi = sigma**2 * mu
        model_params = {
            'xi': xi, 
            'sigma': sigma, 
            'Xc': Xc,
            'dist_type': dist_type, 
            'dist_params': dist_params
        }
        model_params_list.append(model_params)
        panel_titles.append(r'$\xi/\sigma^2 = %.1f$' % mu)
    
    plot_waiting_time_histogram(BrownianGlitchModel, model_params_list, x0, T, N,
                               Nsim=Nsim, panel_titles=panel_titles, seed=seed)