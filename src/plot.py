import numpy as np
import matplotlib.pyplot as plt
from brownian_model import BrownianGlitchModel
from simulation import simulation_parallel

def plot_single_trajectory(mu, xi, Xc, dist_type, dist_params, x0, T, N):
    # Create the system
    system = BrownianGlitchModel(xi=xi, sigma=np.sqrt(xi/mu), Xc=Xc, dist_type=dist_type, dist_params=dist_params)
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

def plot_brownian_glitch_panels(mu_list,
                                xi = None,
                                sigma=None,
                                Xc=1.0, 
                                dist_type='neg_powerlaw', 
                                dist_params={'delta': 1.5, 'beta': 0.01},
                                x0=0.5, 
                                T=50, 
                                N=50_000,
                                figsize=(8,6),
                                title = "process trajectory"):
    """
    Plots a 2x2 panel of BrownianGlitchModel simulations for different xi/sigma^2 values,
    including glitch occurrences as vertical segments at the bottom of each plot.

    Parameters:
        mu_list: list of floats
            List of xi/sigma^2 values to use for each panel (length 4).
        Xc: float
            Critical value for the model.
        dist_type: str
            Distribution type for glitches.
        dist_params: dict
            Parameters for the glitch distribution.
        x0: float
            Initial value.
        T: float
            Total simulation time.
        N: int
            Number of time steps.
        figsize: tuple
            Figure size.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)
    axes = axes.flatten()
    for idx, mu in enumerate(mu_list):
        if xi is None:
            xi = sigma**2 * mu
        else:
            sigma = np.sqrt(xi / mu)
        
        # Create the system
        system = BrownianGlitchModel(Xc=Xc, xi=xi, sigma=sigma, dist_type=dist_type, dist_params=dist_params)
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
    
def plot_waiting_time_histogram(mu_list, 
                          Xc=1.0, 
                          sigma=0.15,
                          dist_type='neg_powerlaw', 
                          dist_params={'delta': 1.5, 'beta': 0.01},
                          x0=0.5, 
                          T=500, 
                          N=500_000,
                          Nsim=100):
    """
    Plots histograms of the waiting times between glitches in a 2x2 subplot layout,
    accumulating results from multiple simulations.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for idx, mu in enumerate(mu_list):
        
        xi = sigma**2 * mu
        
        # Collect waiting times from multiple simulations
        all_waiting_times = []
        for _ in range(Nsim):
            system = BrownianGlitchModel(Xc=Xc, xi=xi, sigma=sigma, dist_type=dist_type, dist_params=dist_params)
            simulation_params = {'x0': x0, 'T': T, 'N': N}
            result = system.simulate(**simulation_params)
            all_waiting_times.extend(result['waiting_times'])
        
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
                                  bins_lin=np.linspace(0.0, 5.0, 50),
                                  bins_log=np.logspace(-2, 2, 50),
                                  sigma=0.15,
                                  system_params=None,
                                  sim_params=None,
                                  N_glitches = 100_000
                                  ):
    """
    Plots waiting time distributions for different mu values in both linear and log scales.
    
    Parameters:
    -----------
    mu_vals : list
        List of mu values to plot
    colormap : list
        List of colors for each mu value
    bins_lin : array
        Linear bins for left panel
    bins_log : array
        Logarithmic bins for right panel
    sigma : float
        Diffusion coefficient
    system_params : dict
        Additional system parameters (optional)
    sim_params : dict
        Additional simulation parameters (optional)
    """
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))

    for mu, c in zip(mu_vals, colormap):
        # Collect at least 20k events
        dt_list = []
        while sum(len(x) for x in dt_list) < N_glitches:
            xi = sigma**2 * mu
            if system_params is None:
                system_params = {}
            
            system_params = {
                'xi': xi, 
                'sigma': sigma, 
                'Xc': system_params.get('Xc', 1.0),
                'dist_type': system_params.get('dist_type', 'neg_powerlaw'),
                'dist_params': system_params.get('dist_params', {'delta': 1.5, 'beta': 1e-2})
            }
            if sim_params is None:
                sim_params = {'x0': 0.5, 'Tsim': 500,
                            'Nsteps': 500_000, 'Nsim': 500}
            
            results = simulation_parallel(system_params, **sim_params)
            for res in results:
                if res is not None:
                    dt_list.append(res['waiting_times'])
                    
        delta_t_tot = np.concatenate(dt_list)[:N_glitches]
        delta_t_norm = delta_t_tot / delta_t_tot.mean()

        # Left panel (log-linear)
        pdf_lin, edges = np.histogram(delta_t_norm, bins=bins_lin, density=True)
        centers_lin = 0.5*(edges[1:] + edges[:-1])
        ax[0].plot(centers_lin, pdf_lin, lw=1.4, color=c, label=fr'$\mu={mu}$')

        # Right panel (log-log)
        pdf_log, edges = np.histogram(delta_t_norm, bins=bins_log, density=True)
        centers_log = np.sqrt(edges[:-1] * edges[1:])  # geometric center
        ax[1].plot(centers_log, pdf_log, lw=1.4, color=c)

    # Left panel formatting
    ax[0].set_yscale('log')
    ax[0].set_xlabel(r'$\Delta t / \langle \Delta t \rangle$')
    ax[0].set_ylabel(r'$p(\Delta t)$')
    ax[0].legend(frameon=False, fontsize=9)
    ax[0].set_xlim(0, 5)
    ax[0].set_ylim(1e-2, 1e1)
    ax[0].set_title('log-linear')

    # Right panel formatting
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xlabel(r'$\Delta t / \langle \Delta t \rangle$')
    ax[1].set_xlim(1e-2, 1e2)
    ax[1].set_ylim(1e-5, 1e2)
    ax[1].set_title('log-log')

    plt.tight_layout()
    plt.show()