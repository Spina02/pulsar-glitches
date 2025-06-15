import numpy as np
from model import Model

class BrownianGlitchModel(Model):
    """
    This class implements an accumulative stochastic model with drift and diffusion,
    threshold Xc and random reset (glitch).
    """

    def __init__(self,
                Xc: float = 1.0,
                xi: float = 0.025,
                sigma: float = 0.15,
                dist_type: str = None,
                dist_params: dict = None,
                seed: int = None,
                only_waits: bool = False
                ) -> None:
        """
        Args:
            Xc (float): threshold.
            xi (float): drift coefficient.
            sigma (float): diffusion coefficient.
            dist_type (str): distribution type.
            dist_params (dict): distribution parameters.
            seed (int, optional): random number generator seed.
        """
        
        super().__init__(Xc=Xc, dist_type=dist_type, dist_params=dist_params, seed=seed)
        
        # Check that the parameters are positive
        if xi <= 0:
            raise ValueError("Drift coefficient xi must be positive.")
        if sigma <= 0:
            raise ValueError("Diffusion coefficient sigma must be positive.")

        # Store the parameters
        self.xi = xi
        self.sigma = sigma
        
        # parameter to optimize the simulation
        self.only_waits = only_waits

    def simulate(self, x0: float, T: float, N: int) -> dict:
        """
        Given the initial population value x0, the considered interval lenght T
        and the number of step in the computation N, this method will return a
        trajectory for the PLS.
        """
        #Check the inputs
        self._Model__checkInputs(x0,T,N)
        
        #Setup step lenght and traj array
        h = T/N
        sqrt_h = np.sqrt(h)
        traj = np.zeros(N+1,dtype=float)
        times = np.zeros(N+1,dtype=float)
        
        # local copy of the parameters to optimize the simulation
        xi = self.xi
        Xc = self.Xc
        
        traj[0] = x0
        times[0] = 0.0
        
        # define storing variables
        glitch_times, waiting_times, glitch_sizes = [0], [], []
        
        # Pre-compute the random steps to avoid repeated calls to rng.normal
        random_steps = self.rng.normal(size=N) * self.sigma * sqrt_h
    
        for i in range(1,N+1):
            times[i] = i*h
                
            # deterministic + stochastic steps
            dx = xi*h + random_steps[i - 1]
            
            # Update the stress level
            x = traj[i-1] + dx
            
            # Reset the total glitch size
            total_glitch_size = 0
            
            # after a glitch, the stress could still be above the threshold -> while loop
            while x - total_glitch_size >= Xc:
                total_glitch_size += max(self._sample_glitch(),0) # avoid negative glitches
                
            traj[i] = max(x - total_glitch_size, 0) # reset step

            # If there was a glitch, store the time and size
            if total_glitch_size > 0:
                if self.only_waits:
                    waiting_times.append(times[i] - glitch_times[-1])
                glitch_sizes.append(total_glitch_size)
                glitch_times.append(times[i])

        # Return the results
        if self.only_waits:
            return np.array(waiting_times)
        return {
                "times": times,
                "traj": traj,
                "glitch_times": np.array(glitch_times[1:]),
                "waiting_times": np.array(waiting_times),
                "glitch_sizes":  np.array(glitch_sizes)
            }
