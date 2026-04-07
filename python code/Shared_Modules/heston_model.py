"""
Heston Stochastic Volatility Model Implementation
"""
import numpy as np
from scipy.integrate import quad

class HestonModel:
    """
    Heston (1993) Stochastic Volatility Model
    """
    
    def __init__(self, kappa, theta, sigma, rho, v0):
        """
        Initialize Heston model parameters
        
        Parameters:
        -----------
        kappa : float - Mean reversion speed
        theta : float - Long-term variance
        sigma : float - Volatility of variance
        rho : float - Correlation between asset and variance
        v0 : float - Initial variance
        """
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.v0 = v0
        
    def characteristic_function(self, u, S0, r, T):
        """
        Lewis (2001) characteristic function for Heston model
        
        Parameters:
        -----------
        u : complex - Integration variable
        S0 : float - Current asset price
        r : float - Risk-free rate
        T : float - Time to maturity
        """
        i = 1j
        
        # Compute d and g
        d = np.sqrt((self.rho * self.sigma * u * i - self.kappa)**2 + 
                    self.sigma**2 * (u * i + u**2))
        g = (self.kappa - self.rho * self.sigma * u * i - d) / \
            (self.kappa - self.rho * self.sigma * u * i + d)
        
        # C and D components
        C = (self.kappa * self.theta / self.sigma**2) * \
            ((self.kappa - self.rho * self.sigma * u * i - d) * T - 
             2 * np.log((1 - g * np.exp(-d * T)) / (1 - g)))
        D = (self.kappa - self.rho * self.sigma * u * i - d) / self.sigma**2 * \
            ((1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T)))
        
        # Characteristic function
        psi = i * u * (np.log(S0) + r * T) + C + D * self.v0
        
        return np.exp(psi)
    
    def call_price_lewis(self, K, S0, r, T):
        """
        Lewis (2001) call option pricing formula
        
        Parameters:
        -----------
        K : float - Strike price
        S0 : float - Current asset price
        r : float - Risk-free rate
        T : float - Time to maturity
        """
        def integrand(u):
            phi = self.characteristic_function(u - 0.5j, S0, r, T)
            k = np.log(K / S0)
            return np.real(np.exp(-1j * u * k) * phi) / (u**2 + 0.25)
        
        # Numerical integration
        integral, _ = quad(integrand, 0, 100, limit=1000)
        
        call_price = S0 - np.sqrt(S0 * K) / np.pi * integral
        return max(call_price, 1e-6)
    
    def put_price_put_call_parity(self, K, S0, r, T):
        """
        Put option price using put-call parity
        """
        call_price = self.call_price_lewis(K, S0, r, T)
        put_price = call_price + K * np.exp(-r * T) - S0
        return max(put_price, 1e-6)
    
    def simulate_paths(self, S0, r, T, n_paths, n_steps):
        """
        Simulate Heston paths using Euler discretization
        
        Parameters:
        -----------
        S0 : float - Initial price
        r : float - Risk-free rate
        T : float - Time horizon
        n_paths : int - Number of paths
        n_steps : int - Number of time steps
        
        Returns:
        --------
        S : array (n_paths, n_steps+1) - Price paths
        v : array (n_paths, n_steps+1) - Variance paths
        """
        dt = T / n_steps
        
        S = np.zeros((n_paths, n_steps + 1))
        v = np.zeros((n_paths, n_steps + 1))
        
        S[:, 0] = S0
        v[:, 0] = self.v0
        
        # Cholesky decomposition for correlated Brownian motions
        corr_matrix = np.array([[1, self.rho], [self.rho, 1]])
        L = np.linalg.cholesky(corr_matrix)
        
        for t in range(n_steps):
            # Generate correlated random numbers
            Z = np.random.normal(0, 1, (n_paths, 2))
            Z_corr = Z @ L.T
            
            # Ensure non-negative variance
            v_half = np.maximum(v[:, t], 0)
            
            # Variance process
            v[:, t + 1] = v[:, t] + self.kappa * (self.theta - v_half) * dt + \
                          self.sigma * np.sqrt(v_half * dt) * Z_corr[:, 1]
            v[:, t + 1] = np.maximum(v[:, t + 1], 0)
            
            # Price process (log transformation for stability)
            S[:, t + 1] = S[:, t] * np.exp((r - 0.5 * v_half) * dt + \
                                            np.sqrt(v_half * dt) * Z_corr[:, 0])
        
        return S, v
    
    def get_parameters(self):
        """Return parameters as dictionary"""
        return {
            'kappa': self.kappa,
            'theta': self.theta,
            'sigma': self.sigma,
            'rho': self.rho,
            'v0': self.v0
        }