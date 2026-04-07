"""
Monte Carlo simulation utilities
"""
import numpy as np

class MonteCarloUtils:
    """Utility class for Monte Carlo simulations"""
    
    @staticmethod
    def price_asian_call(model, S0, r, T, K, n_paths, n_steps, include_initial=True):
        """
        Price Asian call option using Monte Carlo
        
        Parameters:
        -----------
        model : HestonModel or BatesModel
        S0 : float - Initial price
        r : float - Risk-free rate
        T : float - Time to maturity
        K : float - Strike price
        n_paths : int - Number of paths
        n_steps : int - Number of time steps
        include_initial : bool - Include S0 in average
        
        Returns:
        --------
        price : float - Option price
        std_error : float - Standard error
        ci_lower : float - 95% confidence interval lower bound
        ci_upper : float - 95% confidence interval upper bound
        """
        S_paths, _ = model.simulate_paths(S0, r, T, n_paths, n_steps)
        
        if include_initial:
            S_with_initial = np.column_stack([np.ones(n_paths) * S0, S_paths[:, 1:]])
        else:
            S_with_initial = S_paths[:, 1:]
        
        avg_prices = np.mean(S_with_initial, axis=1)
        payoffs = np.maximum(avg_prices - K, 0)
        
        price = np.exp(-r * T) * np.mean(payoffs)
        std_error = np.std(payoffs) / np.sqrt(n_paths)
        ci_lower = price - 1.96 * std_error
        ci_upper = price + 1.96 * std_error
        
        return price, std_error, ci_lower, ci_upper
    
    @staticmethod
    def price_put_option(model, S0, r, T, K, n_paths, n_steps):
        """
        Price European put option using Monte Carlo
        """
        S_paths, _ = model.simulate_paths(S0, r, T, n_paths, n_steps)
        S_T = S_paths[:, -1]
        
        payoffs = np.maximum(K - S_T, 0)
        price = np.exp(-r * T) * np.mean(payoffs)
        std_error = np.std(payoffs) / np.sqrt(n_paths)
        ci_lower = price - 1.96 * std_error
        ci_upper = price + 1.96 * std_error
        
        return price, std_error, ci_lower, ci_upper
    
    @staticmethod
    def convergence_analysis(model, S0, r, T, K, path_counts, n_steps, option_type='asian'):
        """
        Analyze convergence with increasing number of paths
        """
        prices = []
        errors = []
        
        for n_paths in path_counts:
            if option_type == 'asian':
                price, std_err, _, _ = MonteCarloUtils.price_asian_call(
                    model, S0, r, T, K, n_paths, n_steps
                )
            else:
                price, std_err, _, _ = MonteCarloUtils.price_put_option(
                    model, S0, r, T, K, n_paths, n_steps
                )
            prices.append(price)
            errors.append(std_err)
        
        return prices, errors