import numpy as np
from model import Model

class SDPModel(Model):
    """
    This class implements an accumulative stochastic model with drift and diffusion,
    threshold Xc and random reset (glitch).
    """

    def __init__(self,
                alpha: float = 1,
                f: float = 1,
                Xc: float = 1.0,
                dist_type: str = None,
                dist_params: dict = None,
                only_waits: bool = False,
                seed: int = None) -> None:
        """
        Args:
            alpha (float): Parameter controlling the glitch rate λ.
            f (float): The deterministic stress loading rate (dX/dt).
            Xc (float): The critical stress threshold.
            dist_type (str): The distribution type for glitch sizes.
            dist_params (dict): Parameters for the glitch size distribution.
            seed (int, optional): Seed for the random number generator.
        """
        super().__init__(Xc = Xc, dist_type = dist_type, dist_params = dist_params, seed = seed)
        
        # Check that the parameters are positive
        if alpha <= 0:
            raise ValueError("alpha must be positive.")
        if f <= 0:
            raise ValueError("Loading rate f must be positive.")
        
        self.only_waits = only_waits
        
        # Store the parameters
        self.alpha = alpha
        self.f = f

    def _sample_waiting_time(self, x0: float) -> float:
        """
        This method will sample the waiting time for the SDP model.
        """
        # We should use u' = 1 - u for the inverse transform sampling formula, 
        # but since both u and (1-u) are uniform in [0,1], we can use u directly 
        # for computational efficiency
        u = self.rng.random() 
        if u <= 1e-9:
            u = 1e-9
            
        # From Eq. 14 of Fulgenzi et al. (2017)
        return (self.Xc - x0) / self.f * (1 - u**(self.f / self.alpha))
    
    def simulate(self, x0: float, T: float, N: int = None) -> dict:
        '''
        Given the initial population value x0, the considered interval lenght T
        and the number of step in the computation N, this method will return a
        trajectory for the PLS.
        '''
        
        # N not used, but it's here for compatibility with the other models
        
        #Check the inputs
        self._Model__checkInputs(x0,T,1)
        
        #Check the inputs
        if x0 < 0 or x0 >= self.Xc:
            raise ValueError(f"Initial value x0 must be in [0, {self.Xc}).")
        
        #Setup times and traj array
        traj = [x0]
        times = [0.0]
        
        # define storing variables
        glitch_times, waiting_times, glitch_sizes = [0], [], []
        
        current_time = 0
        current_stress = x0
        
        while current_time < T:
            # 1. Sample the waiting time to the next event
            dt = self._sample_waiting_time(current_stress)
            
            # 2. Calculate time and stress just before the next glitch
            t_glitch = current_time + dt
            
            # If the next glitch is outside our simulation window, we stop
            if t_glitch > T:
                times.append(T)
                traj.append(current_stress + self.f * (T - current_time))
                break
            
            x_pre_glitch = min(current_stress + self.f * dt, self.Xc)
            
            # 3. Sample the glitch size, conditional on the pre-glitch stress
            # this ensures that the final stress is always below the threshold
            glitch_size = self._sample_glitch(y=x_pre_glitch)
            
            # 4. Update the current stress
            x_post_glitch = max(x_pre_glitch - glitch_size, 0) # no negative glitches
            
            # Store trajectory points to show the sawtooth pattern
            times.append(t_glitch)
            traj.append(x_pre_glitch)
            times.append(t_glitch) # Same time, different stress value: jump due to glitch
            traj.append(x_post_glitch)

            # Store statistics
            glitch_times.append(t_glitch)
            waiting_times.append(dt)
            glitch_sizes.append(glitch_size)

            # 5. Update state for the next iteration
            current_time = t_glitch
            current_stress = x_post_glitch
        
        if self.only_waits:
            return np.array(waiting_times)
        return {
                "times": np.array(times),
                "traj": np.array(traj),
                "glitch_times": np.array(glitch_times[1:]),
                "waiting_times": np.array(waiting_times),
                "glitch_sizes":  np.array(glitch_sizes)
            }
