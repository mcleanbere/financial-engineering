"""
Sub-group 3: Asian Option Pricing via Monte Carlo
Members 7-10
20-day ATM Asian call option
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Shared_Modules.heston_model import HestonModel
from Shared_Modules.monte_carlo_utils import MonteCarloUtils
from Shared_Modules.visualization_utils import VisualizationUtils

# Constants
S0 = 232.90
r = 0.015
TRADING_DAYS = 250
T = 20 / TRADING_DAYS  # 20 days to years
K = S0  # ATM strike
FEE_PERCENT = 0.04  # 4% bank fee

# Monte Carlo parameters
MC_PATHS = 100000
MC_STEPS = 250  # Daily steps

# Heston parameters (from Sub-group 1 calibration)
HESTON_PARAMS = {
    'kappa': 2.45,
    'theta': 0.043,
    'sigma': 0.58,
    'rho': -0.73,
    'v0': 0.041
}

# Create output directory
output_dir = 'Outputs'
os.makedirs(output_dir, exist_ok=True)

print("=" * 80)
print("SUB-GROUP 3: Asian Option Pricing via Monte Carlo")
print("Product: ATM Asian Call Option")
print(f"Maturity: 20 days")
print(f"Strike: ${K:.2f} (ATM)")
print("=" * 80)

# Initialize Heston model
heston_model = HestonModel(**HESTON_PARAMS)

print(f"\nHeston Model Parameters:")
for key, value in HESTON_PARAMS.items():
    print(f"  {key}: {value}")

# Price Asian option
print(f"\nRunning Monte Carlo simulation with {MC_PATHS:,} paths...")
asian_price, std_error, ci_lower, ci_upper = MonteCarloUtils.price_asian_call(
    heston_model, S0, r, T, K, MC_PATHS, MC_STEPS, include_initial=True
)

# Add bank fee
final_price = asian_price * (1 + FEE_PERCENT)

print("\n" + "=" * 60)
print("ASIAN OPTION PRICING RESULTS")
print("=" * 60)
print(f"Fair Price: ${asian_price:.4f}")
print(f"Bank Fee ({FEE_PERCENT*100:.0f}%): ${asian_price * FEE_PERCENT:.4f}")
print(f"Final Client Price: ${final_price:.4f}")
print(f"95% Confidence Interval: [${ci_lower:.4f}, ${ci_upper:.4f}]")
print(f"Standard Error: ${std_error:.6f}")

# Save results to file
with open(os.path.join(output_dir, 'asian_option_price.txt'), 'w') as f:
    f.write("Asian Call Option Pricing Results\n")
    f.write("=" * 40 + "\n")
    f.write(f"Product: ATM Asian Call Option\n")
    f.write(f"Underlying: SM Energy Company\n")
    f.write(f"Current Price: ${S0:.2f}\n")
    f.write(f"Strike Price: ${K:.2f} (ATM)\n")
    f.write(f"Maturity: 20 days\n")
    f.write(f"Risk-free Rate: {r*100:.2f}%\n\n")
    f.write(f"Monte Carlo Parameters:\n")
    f.write(f"  Number of Paths: {MC_PATHS:,}\n")
    f.write(f"  Number of Steps: {MC_STEPS}\n\n")
    f.write(f"Results:\n")
    f.write(f"  Fair Price: ${asian_price:.4f}\n")
    f.write(f"  Bank Fee ({FEE_PERCENT*100:.0f}%): ${asian_price * FEE_PERCENT:.4f}\n")
    f.write(f"  Final Client Price: ${final_price:.4f}\n")
    f.write(f"  95% Confidence Interval: [${ci_lower:.4f}, ${ci_upper:.4f}]\n")
    f.write(f"  Standard Error: ${std_error:.6f}\n")

# Convergence analysis
path_counts = [1000, 5000, 10000, 50000, 100000]
prices, errors = MonteCarloUtils.convergence_analysis(
    heston_model, S0, r, T, K, path_counts, MC_STEPS, option_type='asian'
)

convergence_df = pd.DataFrame({
    'Paths': path_counts,
    'Price': prices,
    'Std_Error': errors
})
convergence_df.to_csv(os.path.join(output_dir, 'convergence_analysis.csv'), index=False)

print(f"\nConvergence Analysis:")
print(f"{'Paths':<12} {'Price':<12} {'Std Error':<12}")
print("-" * 36)
for n, p, e in zip(path_counts, prices, errors):
    print(f"{n:<12} ${p:<11.4f} ${e:<11.6f}")

# Plot convergence
fig1 = VisualizationUtils.plot_convergence(path_counts, prices, asian_price)
VisualizationUtils.save_figure(fig1, os.path.join(output_dir, 'convergence_analysis.png'))

# Simulate and plot sample paths
S_paths, v_paths = heston_model.simulate_paths(S0, r, T, 1000, MC_STEPS)

fig2 = VisualizationUtils.plot_sample_paths(S_paths, n_samples=20)
fig2.suptitle('Sample Heston Price Paths (20-day Horizon)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'sample_paths.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot price distribution
avg_prices = []
for i in range(10000):  # Use subset for distribution
    S_paths_sub, _ = heston_model.simulate_paths(S0, r, T, 1, MC_STEPS)
    avg = np.mean(np.append([S0], S_paths_sub[0, 1:]))
    avg_prices.append(avg)

fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.hist(avg_prices, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
ax3.axvline(x=K, color='red', linestyle='--', linewidth=2, label=f'Strike: ${K:.2f}')
ax3.axvline(x=asian_price, color='green', linestyle='--', linewidth=2, label=f'Fair Price: ${asian_price:.4f}')
ax3.set_xlabel('Average Stock Price ($)')
ax3.set_ylabel('Probability Density')
ax3.set_title('Distribution of Average Stock Prices (20-day Asian Option)')
ax3.legend()
ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'price_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()

# Save simulation results
simulation_df = pd.DataFrame({
    'Path': range(len(avg_prices)),
    'Average_Price': avg_prices,
    'Payoff': np.maximum(np.array(avg_prices) - K, 0)
})
simulation_df.to_csv(os.path.join(output_dir, 'simulation_results.csv'), index=False)

print("\n" + "=" * 60)
print("OUTPUT FILES GENERATED:")
print(f"1. {output_dir}/asian_option_price.txt - Pricing results")
print(f"2. {output_dir}/convergence_analysis.png - Convergence plot")
print(f"3. {output_dir}/sample_paths.png - Sample paths")
print(f"4. {output_dir}/price_distribution.png - Price distribution")
print(f"5. {output_dir}/simulation_results.csv - Simulation data")
print("=" * 60)

print("\nAsian option pricing completed successfully!")