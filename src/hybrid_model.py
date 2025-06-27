import numpy as np
from model import Model

class HybridModel(Model):
    """
    This class implements a hybrid model that combines the SDP model with the Brownian model.
    """

    def __init__(self,
                alpha: float = 1.0,
                xi: float = 0.025,
                sigma: float = 0.15,
                Xc: float = 1.0,
                dist_type: str = None,
                dist_params: dict = None,
                only_waits: bool = False,
                seed: int = None) -> None:
        """
        Args:
            alpha (float): Parameter controlling the glitch rate λ. It represents (I_c * lambda_0)/N_em.
            xi (float): drift coefficient.
            sigma (float): diffusion coefficient.
            Xc (float): threshold.
            dist_type (str): distribution type.
            dist_params (dict): distribution parameters.
            only_waits (bool): if True, only the waiting times are returned.
            seed (int, optional): random number generator seed.
        """
        
        super().__init__(Xc=Xc, dist_type=dist_type, dist_params=dist_params, seed=seed)
        
        # Check that the parameters are positive
        if alpha <= 0:
            raise ValueError("alpha must be positive.")
        if xi <= 0:
            raise ValueError("Drift coefficient xi must be positive.")
        if sigma <= 0:
            raise ValueError("Diffusion coefficient sigma must be positive.")
        
        # Store the parameters
        self.alpha = alpha*self.Xc
        self.xi = xi
        self.sigma = sigma
        
        # parameter to optimize the simulation
        self.only_waits = only_waits
        
    def _get_lambda(self, x: float) -> float:
        if self.Xc - x <= 1e-9:
            return np.inf
        if x < 0:
            return 0.0
        return self.alpha / (self.Xc - x)
    
    def simulate(self, x0: float, T: float, N: int) -> dict:
        '''
        Given the initial population value x0, the considered interval lenght T
        and the number of step in the computation N, this method will return a
        trajectory for the PLS.
        '''
        #Check the inputs
        self._Model__checkInputs(x0,T,N)
        
        #Setup step lenght and traj array
        h = T/N
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
        random_steps = self.rng.normal(size=N) * self.sigma * np.sqrt(h)
        
        for i in range(1,N+1):
            next_time = i*h
            times[i] = next_time
            
            # 1. Deterministic + stochastic steps
            dx = xi*h + random_steps[i - 1]
            
            # current stress level
            x = traj[i-1] + dx
            
            rate = self._get_lambda(x)
            
            glitch_size = 0
            if x >= self.Xc or self.rng.random() <= rate*h:
                glitch_size = max(self._sample_glitch(y=min(x,self.Xc)),0)
                
            if glitch_size > 0:
                x = max(x - glitch_size, 0) # no negative glitches
            
                # 5. Store statistics
                waiting_times.append(next_time - glitch_times[-1])
                glitch_times.append(next_time)
                glitch_sizes.append(glitch_size)
                
            # 6. Update the stress level
            traj[i] = x
        
        if self.only_waits:
            return np.array(waiting_times)
        return {
                "times": np.array(times),
                "traj": np.array(traj),
                "glitch_times": np.array(glitch_times[1:]),
                "waiting_times": np.array(waiting_times),
                "glitch_sizes":  np.array(glitch_sizes)
            }
