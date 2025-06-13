import numpy as np
from numpy.random import Generator, PCG64
from model import Model

class SDPModel(Model):
    """
    This class implements an accumulative stochastic model with drift and diffusion,
    threshold Xc and random reset (glitch).
    """

    def __init__(self,
                alpha: float = 1,
                dist_type: str = None,
                dist_params: dict = None,
                seed: int = None) -> None:
        """
        Args:
            xi (float): drift coefficient.
            sigma (float): diffusion coefficient.
            Xc (float): threshold.
            dist_type (str): distribution type.
            dist_params (dict): distribution parameters.
            seed (int, optional): random number generator seed.
        """
        super().__init__(dist_type, dist_params, seed)
        
        # Check that the parameters are positive
        if alpha <= 0:
            raise ValueError("alpha must be positive.")
        
        # Store the parameters
        self.alpha = alpha
        
    def get_lambda(self, x: float) -> float:
        """
        This method will compute the lambda parameter for the SDP model.
        """
        if self.Xc - x <= 1e-9:
            return np.inf
        return self.alpha / (self.Xc-x)
    
    def _sample_waiting_time(self, x: float) -> float:
        """
        This method will sample the waiting time for the SDP model.
        """
        u = self.rng.random()
        if u <= 1e-9:
            u = 1e-9
        return (self.Xc - x) * (1 - u**(1/self.alpha))
    
    def simulate(self, x0: float, T: float, N: int) -> np.ndarray:
        '''
        Given the initial population value x0, the considered interval lenght T
        and the number of step in the computation N, this method will return a
        trajectory for the PLS.
        '''
        #Check the inputs
        self.__checkInputs(x0,T,N)
        
        #Setup step lenght and traj array
        h = T/N
        traj = np.zeros(N+1,dtype=float)
        times = np.zeros(N+1,dtype=float)
        
        traj[0] = x0
        times[0] = 0.0
        
        # define storing variables
        glitch_times, waits, sizes = [0], [], []
        
        #TODO: Implement the simulation
        last_glitch_time = 0
        wait = self._sample_waiting_time(traj[0])
        waits.append(wait)
        for i in range(1,N+1):
            times[i] = i*h
            
            x = traj[i-1] + self.get_lambda(traj[i-1])*h
            
            total_glitch_size = 0
            if times[i] > last_glitch_time + wait:
                glitch_times.append(times[i])
                
                while x - total_glitch_size >= self.Xc:
                    total_glitch_size += max(self._sample_glitch(),0)
                sizes.append(total_glitch_size)
                last_glitch_time = times[i]
                wait = self._sample_waiting_time(x + total_glitch_size)
                
            traj[i] = max(x - total_glitch_size, 0)

        return {
                "times": times,
                "traj": traj,
                "glitch_times": np.array(glitch_times[1:]),
                "waiting_times": np.array(waits),
                "glitch_sizes":  np.array(sizes)
            }
