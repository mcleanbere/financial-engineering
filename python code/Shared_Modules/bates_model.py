"""
Bates Model Implementation (Heston with Jumps)
"""
import numpy as np
from .heston_model import HestonModel

class BatesModel(HestonModel):
    """
    Bates (1996) Model: Heston with Poisson jumps
    
    Additional parameters:
    - lambd : Jump intensity
    - mu_j : Mean jump size
    - sigma_j : Jump volatility
    """
    
    def __init__(self, kappa, theta, sigma, rho, v0, lambd, mu_j, sigma_j):
        super().__init__(kappa, theta, sigma, rho, v0)
        self.lambd = lambd
        self.mu_j = mu_j
        self.sigma_j = sigma_j
        
    def jump_characteristic_function(self, u, T):
        """
        Jump component characteristic function
        """
        i = 1j
        return np.exp(self.lambd * T * 
                     (np.exp(i * u * self.mu_j - 0.5 * u**2 * self.sigma_j**2) - 1))
    
    def characteristic_function(self, u, S0, r, T):
        """
        Bates characteristic function = Heston CF × Jump CF
        """
        heston_cf = super().characteristic_function(u, S0, r, T)
        jump_cf = self.jump_characteristic_function(u, T)
        return heston_cf * jump_cf
    
    def call_price_lewis(self, K, S0, r, T):
        """
        Lewis pricing with jumps
        """
        from scipy.integrate import quad
        
        def integrand(u):
            phi = self.characteristic_function(u - 0.5j, S0, r, T)
            k = np.log(K / S0)
            return np.real(np.exp(-1j * u * k) * phi) / (u**2 + 0.25)
        
        integral, _ = quad(integrand, 0, 100, limit=1000)
        call_price = S0 - np.sqrt(S0 * K) / np.pi * integral
        return max(call_price, 1e-6)
    
    def simulate_paths(self, S0, r, T, n_paths, n_steps):
        """
        Simulate Bates model paths with jumps
        """
        dt = T / n_steps
        
        S = np.zeros((n_paths, n_steps + 1))
        v = np.zeros((n_paths, n_steps + 1))
        
        S[:, 0] = S0
        v[:, 0] = self.v0
        
        # Correlation matrix
        corr_matrix = np.array([[1, self.rho], [self.rho, 1]])
        L = np.linalg.cholesky(corr_matrix)
        
        # Jump compensation term
        jump_comp = self.lambd * (np.exp(self.mu_j + 0.5 * self.sigma_j**2) - 1)
        
        for t in range(n_steps):
            # Generate correlated random numbers
            Z = np.random.normal(0, 1, (n_paths, 2))
            Z_corr = Z @ L.T
            
            v_half = np.maximum(v[:, t], 0)
            
            # Variance process
            v[:, t + 1] = v[:, t] + self.kappa * (self.theta - v_half) * dt + \
                          self.sigma * np.sqrt(v_half * dt) * Z_corr[:, 1]
            v[:, t + 1] = np.maximum(v[:, t + 1], 0)
            
            # Jump component
            n_jumps = np.random.poisson(self.lambd * dt, n_paths)
            jump = np.zeros(n_paths)
            for i in range(n_paths):
                if n_jumps[i] > 0:
                    jump[i] = np.sum(np.random.lognormal(self.mu_j, self.sigma_j, n_jumps[i]) - 1)
            
            # Price process with jumps
            S[:, t + 1] = S[:, t] * (np.exp((r - jump_comp - 0.5 * v_half) * dt + \
                                            np.sqrt(v_half * dt) * Z_corr[:, 0]) + jump)
        
        return S, v
    
    def get_parameters(self):
        """Return all parameters as dictionary"""
        params = super().get_parameters()
        params.update({
            'lambd': self.lambd,
            'mu_j': self.mu_j,
            'sigma_j': self.sigma_j
        })
        return params