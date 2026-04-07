"""
Cox-Ingersoll-Ross (1985) Interest Rate Model
"""
import numpy as np

class CIRModel:
    """
    CIR interest rate model: dr = a(b - r)dt + σ√r dW
    """
    
    def __init__(self, a, b, sigma):
        self.a = a      # Mean reversion speed
        self.b = b      # Long-term mean
        self.sigma = sigma  # Volatility
        
    def zero_coupon_bond_price(self, r, t, T):
        """
        Closed-form zero-coupon bond price
        
        Parameters:
        -----------
        r : float - Current interest rate
        t : float - Current time
        T : float - Maturity time
        
        Returns:
        --------
        P : float - Bond price
        """
        tau = T - t
        
        gamma = np.sqrt(self.a**2 + 2 * self.sigma**2)
        
        B = 2 * (np.exp(gamma * tau) - 1) / ((gamma + self.a) * (np.exp(gamma * tau) - 1) + 2 * gamma)
        A = ((2 * gamma * np.exp((self.a + gamma) * tau / 2)) / 
             ((gamma + self.a) * (np.exp(gamma * tau) - 1) + 2 * gamma)) ** (2 * self.a * self.b / self.sigma**2)
        
        return A * np.exp(-B * r)
    
    def yield_to_maturity(self, r, t, T):
        """
        Compute yield from bond price
        """
        P = self.zero_coupon_bond_price(r, t, T)
        tau = T - t
        return -np.log(P) / tau
    
    def simulate_paths(self, r0, T, n_paths, n_steps):
        """
        Simulate CIR paths using Euler discretization with reflection
        
        Parameters:
        -----------
        r0 : float - Initial rate
        T : float - Time horizon
        n_paths : int - Number of paths
        n_steps : int - Number of time steps
        
        Returns:
        --------
        rates : array (n_paths, n_steps+1) - Rate paths
        """
        dt = T / n_steps
        rates = np.zeros((n_paths, n_steps + 1))
        rates[:, 0] = r0
        
        for t in range(n_steps):
            r_t = np.maximum(rates[:, t], 0)  # Ensure non-negative
            Z = np.random.normal(0, 1, n_paths)
            
            drift = self.a * (self.b - r_t) * dt
            diffusion = self.sigma * np.sqrt(r_t * dt) * Z
            
            rates[:, t + 1] = r_t + drift + diffusion
            rates[:, t + 1] = np.maximum(rates[:, t + 1], 0)  # Reflection
        
        return rates
    
    def simulate_future_rate(self, r0, T, n_paths, n_steps):
        """
        Simulate and analyze future rate distribution
        """
        rates = self.simulate_paths(r0, T, n_paths, n_steps)
        rates_T = rates[:, -1]
        
        return {
            'paths': rates,
            'final_rates': rates_T,
            'mean': np.mean(rates_T),
            'std': np.std(rates_T),
            'percentile_2.5': np.percentile(rates_T, 2.5),
            'percentile_97.5': np.percentile(rates_T, 97.5)
        }
    
    def get_parameters(self):
        """Return parameters as dictionary"""
        return {
            'a': self.a,
            'b': self.b,
            'sigma': self.sigma
        }