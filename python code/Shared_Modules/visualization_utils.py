"""
Visualization utilities for the project
"""
import numpy as np
import matplotlib.pyplot as plt

class VisualizationUtils:
    """Utility class for creating visualizations"""
    
    @staticmethod
    def plot_calibration_fit(market_strikes, market_prices, model_prices, S0, title):
        """
        Plot calibration fit: market vs model prices
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.scatter(market_strikes, market_prices, color='red', s=50, 
                   alpha=0.7, label='Market Prices')
        ax.plot(market_strikes, model_prices, 'b-', linewidth=2, 
                label='Model Prices')
        ax.axvline(x=S0, color='green', linestyle='--', linewidth=1.5,
                   label=f'Spot: ${S0:.2f}')
        
        ax.set_xlabel('Strike Price ($)', fontsize=12)
        ax.set_ylabel('Option Price ($)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_convergence(path_counts, prices, true_price=None):
        """
        Plot Monte Carlo convergence
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(path_counts, prices, 'bo-', linewidth=2, markersize=8,
                label='Simulated Price')
        
        if true_price is not None:
            ax.axhline(y=true_price, color='r', linestyle='--', linewidth=2,
                       label=f'Converged: ${true_price:.4f}')
        
        ax.set_xlabel('Number of Monte Carlo Paths', fontsize=12)
        ax.set_ylabel('Option Price ($)', fontsize=12)
        ax.set_title('Monte Carlo Convergence Analysis', fontsize=14)
        ax.set_xscale('log')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_term_structure(T_points, rates, spline_T=None, spline_rates=None):
        """
        Plot interest rate term structure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(T_points, rates, 'ro', markersize=8, label='Market Data')
        
        if spline_T is not None and spline_rates is not None:
            ax.plot(spline_T, spline_rates, 'b-', linewidth=2, 
                    label='Cubic Spline Interpolation')
        
        ax.set_xlabel('Time to Maturity (Years)', fontsize=12)
        ax.set_ylabel('Interest Rate (%)', fontsize=12)
        ax.set_title('Euribor Term Structure', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_rate_distribution(rates, mean, ci_lower, ci_upper, title):
        """
        Plot distribution of simulated rates
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(rates * 100, bins=100, density=True, alpha=0.7, 
                color='skyblue', edgecolor='black')
        ax.axvline(mean * 100, color='r', linestyle='--', linewidth=2,
                   label=f'Mean: {mean*100:.2f}%')
        ax.axvline(ci_lower * 100, color='g', linestyle=':', linewidth=2,
                   label='95% Confidence Interval')
        ax.axvline(ci_upper * 100, color='g', linestyle=':', linewidth=2)
        
        ax.set_xlabel('Interest Rate (%)', fontsize=12)
        ax.set_ylabel('Probability Density', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_sample_paths(paths, n_samples=10):
        """
        Plot sample paths from simulation
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        n_steps = paths.shape[1]
        time = np.linspace(0, 1, n_steps)
        
        indices = np.random.choice(paths.shape[0], min(n_samples, paths.shape[0]), replace=False)
        
        for idx in indices:
            ax.plot(time, paths[idx, :], linewidth=0.5, alpha=0.7)
        
        ax.set_xlabel('Time (Years)', fontsize=12)
        ax.set_ylabel('Price / Rate', fontsize=12)
        ax.set_title('Sample Simulation Paths', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def save_figure(fig, filepath, dpi=300):
        """Save figure to file"""
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        plt.close(fig)