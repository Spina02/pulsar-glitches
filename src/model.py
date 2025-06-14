import numpy as np
from numpy.random import Generator, PCG64

class Model:
    def __init__(self,
                Xc: float = 1.0,
                dist_type: str = None,
                dist_params: dict = None,
                seed: int = None) -> None:
        """
        Args:
            Xc (float): threshold.
            dist_type (str): distribution type.
            dist_params (dict): distribution parameters.
            seed (int, optional): random number generator seed.
        """
        # Store the threshold
        if Xc <= 0:
            raise ValueError("Threshold Xc must be positive.")
        self.Xc = Xc
        
        # Distributions parameters
        self.dist_type = dist_type
        self.dist_params = dist_params
        if dist_type == "neg_powerlaw":
            self._beta = self.dist_params.get("beta", 0.01)
            self._delta = self.dist_params.get("delta", 1.5)
            if self._beta <= 0:
                raise ValueError("beta for negative power law must be positive.")
            if self._delta <= 0:
                raise ValueError("delta for negative power law must be positive.")
        elif dist_type == "gaussian":
            self._mean = self.dist_params.get("mean", 0.5)
            self._std = self.dist_params.get("std", 0.125)
            if self._std <= 0:
                raise ValueError("standard deviation for Gaussian must be positive.")
        elif dist_type == "lognormal":
            self._mu = self.dist_params.get("mu", -1)
            self._sigma = self.dist_params.get("sigma", 0.5)
            if self._sigma <= 0:
                raise ValueError("sigma for Lognormal must be positive.")
        else:
            raise ValueError(f"Distribution type '{self.dist_type}' not recognized or implemented.")

        # Random number generator
        self.seed = seed
        self.rng = Generator(PCG64(seed))
    
    def __checkInputs(self, x0: float, T: float, N: int) -> None:
        """
        Check that the inputs are correct.
        """
        if x0 <= 0:
            raise ValueError("Initial value x0 must be positive.")
        if T <= 0:
            raise ValueError("Time T must be positive.")
        if N <= 0:
            raise ValueError("Number of steps N must be positive.")
        
    def _neg_powerlaw_sample(self, beta: float, delta: float, y: float = 1.0) -> float:
        """
        Implements inverse transform sampling for a truncated (negative) power-law 
        distribution X^-delta.

        The sampling is performed on the normalized interval [beta, x_max], where:
        - If y is provided (SDP model case), x_max = y / self.Xc, representing the 
        conditional stress limit.
        - If y is not provided (Brownian model case), x_max = 1.0.

        Args:
            beta (float): The lower cut-off of the normalized distribution.
            delta (float): The power-law index.
            y (float, optional): The conditional pre-glitch stress. Defaults to None.

        Returns:
            float: A normalized sample from the distribution.
        """
        u = self.rng.random()
        x_min = beta
        
        # Set the upper bound of the sampling interval
        x_max = y / self.Xc
        
        # Ensure the physical constraint that x_max must be >= x_min
        if x_max < x_min:
            # This can happen in the SDP model if the pre-glitch stress is very low.
            # In this case, the only possible glitch size is x_max itself.
            return x_max

        # Special case for delta = 1, which leads to a logarithmic form
        if abs(delta - 1.0) < 1e-8:
            # The inverse CDF for this case is: x = x_min^(1-u) * x_max^u
            return (x_min**(1.0 - u)) * (x_max**u)
        
        # Handle the general case for delta != 1
        else:
            # The inverse CDF formula for the truncated power law is:
            # sample = [(x_max^(1-d) - x_min^(1-d))*u + x_min^(1-d)]^(1/(1-d))
            one_minus_delta = 1.0 - delta
            x_min_pow = x_min ** one_minus_delta
            x_max_pow = x_max ** one_minus_delta
            # Calculate the sample using the inverse CDF formula
            sample = (u * (x_max_pow - x_min_pow) + x_min_pow)**(1.0 / one_minus_delta)
            return sample
        
    def _gaussian_sample(self, mean, std, y = 1) -> float:
        y_norm = y/self.Xc
        while True:
            sample = self.rng.normal(mean, std)
            if 0 <= sample <= y_norm:
                return sample
            
    def _lognormal_sample(self, mean: float, std: float, y = 1) -> float:
        """
        Samples from a Lognormal distribution with parameters mean, std for log(X),
        truncated to the normalized interval [0, 1].
        Returns a normalized sample (0, 1].
        """
        y_norm = y/self.Xc
        while True:
            x = self.rng.normal(mean, std)
            sample_norm = np.exp(x)
            if sample_norm <= y_norm: # > 0 is implicit in exp
                return sample_norm
    
    def _sample_glitch(self, y: float = 1) -> float:
        """
        Samples the size of the stress release, delta_X, from a specified distribution.
        The size is normalized by Xc, so it's a value between 0 and 1.
        """
        if self.dist_type == "neg_powerlaw":
            # The sample is delta_X/Xc: we need to multiply by Xc
            return self._neg_powerlaw_sample(self._beta, self._delta, y)*self.Xc 
        elif self.dist_type == "gaussian":
            return self._gaussian_sample(self._mean, self._std, y)*self.Xc 
        elif self.dist_type == "lognormal":
            return self._lognormal_sample(self._mu, self._sigma, y)*self.Xc
        # else not needed: checked already in __init__
        
    def simulate(self, x0: float, T: float) -> dict:
        raise NotImplementedError
    