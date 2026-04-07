"""
Calibration utilities for option pricing models
"""
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.fft import fft, fftfreq
from .heston_model import HestonModel
from .bates_model import BatesModel

class CalibrationUtils:
    """Utility class for model calibration"""
    
    @staticmethod
    def calibrate_heston_lewis(market_data, S0, r, T, initial_params=None):
        """
        Calibrate Heston model using Lewis (2001) method
        """
        if initial_params is None:
            initial_params = [2.0, 0.04, 0.5, -0.7, 0.04]
        
        bounds = [(0.01, 20), (0.001, 1), (0.01, 3), (-0.99, 0.99), (0.001, 1)]
        
        def objective(params):
            kappa, theta, sigma, rho, v0 = params
            model = HestonModel(kappa, theta, sigma, rho, v0)
            
            mse = 0
            for _, row in market_data.iterrows():
                model_price = model.call_price_lewis(row['strike'], S0, r, T)
                mse += (model_price - row['call_price'])**2
            
            return mse / len(market_data)
        
        result = minimize(objective, initial_params, method='L-BFGS-B', bounds=bounds)
        
        if result.success:
            return HestonModel(*result.x), result.fun
        else:
            # Fallback to differential evolution
            result = differential_evolution(objective, bounds, maxiter=100, popsize=15)
            return HestonModel(*result.x), result.fun
    
    @staticmethod
    def calibrate_heston_carrmadan(market_data, S0, r, T, alpha=1.0, N=4096):
        """
        Calibrate Heston model using Carr-Madan (1999) FFT method
        """
        def carr_madan_prices(model, strikes):
            """Compute prices using FFT"""
            eta = 0.25
            lambda_val = 2 * np.pi / (N * eta)
            k = np.linspace(-lambda_val * N/2, lambda_val * N/2, N)
            u = np.arange(1, N + 1) * eta
            
            psi = np.zeros(N, dtype=complex)
            for i, u_val in enumerate(u):
                phi = model.characteristic_function(u_val - (alpha + 1) * 1j, S0, r, T)
                psi[i] = np.exp(-r * T) * phi / (alpha**2 + alpha - u_val**2 + 
                                                  1j * (2*alpha+1) * u_val)
            
            weights = np.ones(N)
            weights[1::2] = 4
            weights[2::2] = 2
            weights[0] = 1
            weights[-1] = 1
            
            fft_vals = fft(psi * weights) * eta / 3
            call_prices = np.exp(-alpha * k) / np.pi * np.real(fft_vals)
            
            results = {}
            for K in strikes:
                k_val = np.log(K / S0)
                idx = np.argmin(np.abs(k - k_val))
                results[K] = max(call_prices[idx], 1e-6)
            
            return results
        
        def objective(params):
            kappa, theta, sigma, rho, v0 = params
            model = HestonModel(kappa, theta, sigma, rho, v0)
            
            strikes = market_data['strike'].values
            model_prices_dict = carr_madan_prices(model, strikes)
            
            mse = 0
            for _, row in market_data.iterrows():
                mse += (model_prices_dict[row['strike']] - row['call_price'])**2
            
            return mse / len(market_data)
        
        bounds = [(0.01, 20), (0.001, 1), (0.01, 3), (-0.99, 0.99), (0.001, 1)]
        result = minimize(objective, [2.0, 0.04, 0.5, -0.7, 0.04], 
                         method='L-BFGS-B', bounds=bounds)
        
        if result.success:
            return HestonModel(*result.x), result.fun
        else:
            result = differential_evolution(objective, bounds, maxiter=100)
            return HestonModel(*result.x), result.fun
    
    @staticmethod
    def calibrate_bates(market_data, S0, r, T, initial_params=None):
        """
        Calibrate Bates model (Heston with jumps)
        """
        if initial_params is None:
            initial_params = [2.5, 0.045, 0.55, -0.72, 0.042, 0.2, -0.02, 0.15]
        
        bounds = [(0.01, 20), (0.001, 1), (0.01, 3), (-0.99, 0.99), (0.001, 1),
                  (0, 5), (-0.5, 0.5), (0.01, 1)]
        
        def objective(params):
            kappa, theta, sigma, rho, v0, lambd, mu_j, sigma_j = params
            model = BatesModel(kappa, theta, sigma, rho, v0, lambd, mu_j, sigma_j)
            
            mse = 0
            for _, row in market_data.iterrows():
                model_price = model.call_price_lewis(row['strike'], S0, r, T)
                mse += (model_price - row['call_price'])**2
            
            return mse / len(market_data)
        
        result = minimize(objective, initial_params, method='L-BFGS-B', bounds=bounds)
        
        if result.success:
            return BatesModel(*result.x), result.fun
        else:
            result = differential_evolution(objective, bounds, maxiter=100, popsize=20)
            return BatesModel(*result.x), result.fun