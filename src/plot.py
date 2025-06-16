import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import os
import pandas as pd

from brownian_model import BrownianGlitchModel
from sdp_model import SDPModel
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


def plot_single_trajectory(mu, xi, Xc, dist_type, dist_params, x0, T, N, seed=None):
    # Create the system
    system = BrownianGlitchModel(xi=xi, sigma=np.sqrt(xi/mu), Xc=Xc, dist_type=dist_type, dist_params=dist_params, seed=seed)
    # Run the simulation
    result = system.simulate(x0, T, N)
    
    # Plot the trajectory
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(result['times'], result['traj'], color='slateblue', lw=1)
    ax.axhline(Xc, color='red', linestyle='--', linewidth=1)
    
    # Plot glitches as vertical lines at the bottom
    glitch_times = result['glitch_times']
    for glitch_time in glitch_times:
        ax.axvline(glitch_time, ymin=0, ymax=0.05, color='black', linewidth=1)
    
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, T)
    ax.set_title(r'$\xi/\sigma^2 = %.1f$' % mu)
    plt.show()


def plot_brownian_glitch_panels(mu_list, xi=None, sigma=None, Xc=1.0, 
                                dist_type='neg_powerlaw', 
                                dist_params={'delta': 1.5, 'beta': 0.01},
                                x0=0.5, T=50, N=50_000,
                                figsize=(8,6), title="process trajectory", seed=None):
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)
    axes = axes.flatten()
    for idx, mu in enumerate(mu_list):
        if xi is None:
            xi = sigma**2 * mu
        else:
            sigma = np.sqrt(xi / mu)
        
        # Create the system
        system = BrownianGlitchModel(Xc=Xc, xi=xi, sigma=sigma, dist_type=dist_type, dist_params=dist_params, seed=seed)
        # Run the simulation
        result = system.simulate(x0, T, N)
        
        ax = axes[idx]
        # Plot trajectory
        ax.plot(result['times'], result['traj'], color='slateblue', lw=1)
        # Plot Xc as dashed red line
        ax.axhline(Xc, color='red', linestyle='--', linewidth=1)
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
        ax.set_title(r'$\xi/\sigma^2 = %.1f$' % mu, fontsize=12)
        if idx % 2 == 0:
            ax.set_ylabel(r'Stress, $X$')
        if idx >= 2:
            ax.set_xlabel(r'Time, $t$ (arb. units)')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_waiting_time_histogram(mu_list, Xc=1.0, sigma=0.15,
                                dist_type='neg_powerlaw', 
                                dist_params={'delta': 1.5, 'beta': 0.01},
                                x0=0.5, T=500, N=500_000,
                                Nsim=100, seed=None):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for idx, mu in enumerate(mu_list):
        
        xi = sigma**2 * mu
        
        system_params = {
            'xi': xi, 'sigma': sigma, 'Xc': Xc,
            'dist_type': dist_type, 'dist_params': dist_params
        }
        
        # Collect waiting times from multiple simulations
        all_waiting_times = []
        
        results = simulation_parallel(system_params, x0, Nsim, Tsim=T, Nsteps=N, only_waits=False)
            
        # Collect waiting times from all simulation results
        all_waiting_times = []
        for res in results:
            if res and 'waiting_times' in res and len(res['waiting_times']) > 0:
                all_waiting_times.extend(res['waiting_times'])
        
        all_waiting_times = np.array(all_waiting_times)
        
        ax = axes[idx]
        ax.hist(all_waiting_times, bins=100, density=True, alpha=0.5)
        ax.set_xlabel(r'Waiting time, $\Delta t$')
        ax.set_ylabel(r'PDF')
        ax.set_title(r'$\xi/\sigma^2 = %.1f$' % mu)
        ax.set_xlim(0, 25)
        ax.set_ylim(0, 0.7)
    
    plt.tight_layout()
    plt.show()


def plot_waiting_time_distributions(mu_vals=[0.1, 1.0, 10.0, 50.0],
                                  colormap=['tab:red', 'tab:orange', 'tab:green', 'tab:blue'],
                                  bins_lin=np.linspace(0.0, 5.0, 40),
                                  bins_log=np.logspace(-2, 2, 40),
                                  sigma=0.15,
                                  system_params=None,
                                  sim_params=None,
                                  N_glitches=100_000,
                                  seed=None,
                                  store_results=False,
                                  save_fig=True,
                                  results_dir="results"
                                  ):
    """
    Plots waiting time distributions, with options to save/load results and figures.
    """
    # Create the results directory if it doesn't exist
    if store_results or save_fig:
        os.makedirs(results_dir, exist_ok=True)
        
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    
    base_rng = np.random.default_rng(seed)

    base_system_params = {
        'sigma': sigma,
        'Xc': system_params.get('Xc', 1.0) if system_params else 1.0,
        'dist_type': system_params.get('dist_type', 'neg_powerlaw') if system_params else 'neg_powerlaw',
        'dist_params': system_params.get('dist_params', {'delta': 1.5, 'beta': 1e-2}) if system_params else {'delta': 1.5, 'beta': 1e-2},
    }

    base_sim_params = {
        'x0': sim_params.get('x0', 0.5) if sim_params else 0.5,
        'Tsim': sim_params.get('Tsim', 500) if sim_params else 500,
        'Nsteps': sim_params.get('Nsteps', 500_000) if sim_params else 500_000,
    }
    
    all_mu_params_for_filename = {}
    
    for mu, c in zip(mu_vals, colormap):
        
        current_system_params = base_system_params.copy()
        current_system_params['xi'] = sigma**2 * mu
        
        current_sim_params = base_sim_params.copy()
        if mu >= 10.0: current_sim_params['Nsteps'] = 10_000_000
        if mu >= 50.0: current_sim_params['Nsteps'] = 50_000_000
        
        # caching results
        params_for_filename = {**current_system_params, **current_sim_params, 'N_glitches': N_glitches}
        filename = _create_filename("waiting_times", params_for_filename)
        filepath = os.path.join(results_dir, filename)
        
        all_mu_params_for_filename[f"mu{mu}".replace('.','p')] = N_glitches # For the figure filename

        if os.path.exists(filepath):
            print(f"Loading cached results for mu = {mu} from {filepath}")
            df = pd.read_csv(filepath)
            delta_t_tot = df['waiting_time'].values
        else:
            print(f"Starting analysis for mu = {mu}...")
            dt_list = []
            collected_glitches = 0
            
            max_sims = N_glitches // 10 if N_glitches > 100 else 100
            seeds = base_rng.integers(low=0, high=2**31, size=max_sims)
            
            def arg_generator():
                for s in seeds:
                    yield (current_sim_params['x0'], current_sim_params['Tsim'], current_sim_params['Nsteps'], s)

            model_init_args = {**current_system_params, 'only_waits': True}

            num_processes = max(1, mp.cpu_count() - 1)
            with mp.Pool(processes=num_processes, initializer=init_worker, initargs=(model_init_args,)) as pool:
                results_iterator = pool.imap_unordered(run_single_simulation_worker, arg_generator())
                for i, waiting_times in enumerate(results_iterator):
                    if waiting_times is not None and len(waiting_times) > 0:
                        dt_list.append(waiting_times)
                        collected_glitches += len(waiting_times)
                    print(f"\r[mu={mu}] Collected {collected_glitches}/{N_glitches} glitches from {i+1} simulations...", end="")
                    if collected_glitches >= N_glitches:
                        pool.terminate()
                        break
            print("\nDone.")
            
            delta_t_tot = np.concatenate(dt_list)[:N_glitches]
            
            if store_results:
                print(f"Storing results to {filepath}")
                pd.DataFrame({'waiting_time': delta_t_tot}).to_csv(filepath, index=False)
        
        # plotting
        if delta_t_tot.mean() > 0:
            delta_t_norm = delta_t_tot / delta_t_tot.mean()
        else:
            delta_t_norm = delta_t_tot

        pdf_lin, edges = np.histogram(delta_t_norm, bins=bins_lin, density=True)
        centers_lin = 0.5 * (edges[1:] + edges[:-1])
        ax[0].plot(centers_lin, pdf_lin, lw=1.4, color=c, label=fr'$\mu={mu}$')

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
        fig_filename_base =f"wtd_plot_{base_system_params['dist_type']}.png"
        fig_filepath = os.path.join(results_dir, fig_filename_base)
        print(f"Saving figure to {fig_filepath}")
        plt.savefig(fig_filepath, dpi=300)
    
    plt.show()


def plot_sdp_vs_brownian(T=2.5, N=5000, seed=None):
    fig, axes = plt.subplots(2, 2, figsize=(10, 5), sharex='col', gridspec_kw={'height_ratios': [1, 0.6]})
    
    # Common parameters for both models
    common_params = {
        'Xc': 1.0,
        'dist_type': 'neg_powerlaw', 
        'dist_params': {'delta': 1.5, 'beta': 1e-2}
    }
    
    # --- Left Panels: SDP Model ---
    sdp_params = {
        'alpha': 1.0, 
        'f': 1.0,
        **common_params
    }
    sdp_model = SDPModel(**sdp_params, seed=seed)
    sdp_result = sdp_model.simulate(x0=0.1, T=T)
    
    # --- Right Panels: Brownian Model ---
    mu = 50.0
    sigma = 0.2
    brownian_params = {
        'sigma': sigma,
        'xi': sigma**2 * mu,
        **common_params
    }
    brownian_model = BrownianGlitchModel(**brownian_params, seed=seed)
    brownian_result = brownian_model.simulate(x0=0.1, T=T, N=N)
    
    # Model results for plotting
    models_data = [
        ('SDP Model', sdp_result, 0),
        ('Brownian Model', brownian_result, 1)
    ]
    
    # Visualization parameters
    v_dot = 0.1  # Arbitrary spin-down rate for visualization
    time_continuous = np.linspace(0, T, 1000)
    baseline_velocity = -v_dot * time_continuous + 0.5
    
    for title, result, col_idx in models_data:
        ax_stress = axes[0, col_idx]
        ax_vel = axes[1, col_idx]
        
        # Top panel: Stress Trajectory
        ax_stress.plot(result['times'], result['traj'], lw=1, color='blue')
        ax_stress.axhline(1.0, color='red', linestyle='--', lw=1, alpha=0.7)
        ax_stress.set_ylim(0, 1.1)
        ax_stress.set_xlim(0, T)
        ax_stress.set_title(title, fontsize=12, pad=10)
        
        # Add glitch markers
        for t in result['glitch_times']:
            ax_stress.axvline(t, ymin=0, ymax=0.05, color='k', lw=1)
        
        # Add cumulative glitch effects
        total_velocity = baseline_velocity.copy()
        for t_glitch, glitch_size in zip(result['glitch_times'], result['glitch_sizes']):
            mask = time_continuous >= t_glitch
            total_velocity[mask] += glitch_size * 2e-2
        ax_vel.plot(time_continuous, total_velocity, color='darkorange', lw=2)

        ax_vel.set_xlabel('Time, t (arb. units)')
        ax_vel.set_xlim(0, T)
        ax_vel.set_ylim(0.2, 0.6)
    
    # Set labels for left column only
    axes[0, 0].set_ylabel('Stress, X')
    axes[1, 0].set_ylabel('Angular velocity\n(arb. units)')

    fig.tight_layout()
    plt.show()