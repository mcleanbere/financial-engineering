"""
Sub-group 3: Put Option Pricing
Members 7-10
70-day Put Option (95% Moneyness)
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Shared_Modules.bates_model import BatesModel
from Shared_Modules.monte_carlo_utils import MonteCarloUtils
from Shared_Modules.visualization_utils import VisualizationUtils

# Constants
S0 = 232.90
r = 0.015
TRADING_DAYS = 250
T = 70 / TRADING_DAYS  # 70 days to years
MONEYNESS = 0.95
K = S0 * MONEYNESS  # Strike price

# Bates parameters (from calibration in Step 2)
BATES_PARAMS = {
    'kappa': 2.81,
    'theta': 0.047,
    'sigma': 0.54,
    'rho': -0.71,
    'v0': 0.044,
    'lambd': 0.23,
    'mu_j': -0.018,
    'sigma_j': 0.15
}

# Monte Carlo parameters
MC_PATHS = 100000
MC_STEPS = 250

# Create output directory
output_dir = 'Outputs'
os.makedirs(output_dir, exist_ok=True)

print("=" * 80)
print("SUB-GROUP 3: Put Option Pricing")
print("Product: European Put Option (95% Moneyness)")
print(f"Maturity: 70 days")
print(f"Strike: ${K:.2f} ({MONEYNESS*100:.0f}% of spot)")
print("=" * 80)

# Initialize Bates model
bates_model = BatesModel(**BATES_PARAMS)

print(f"\nBates Model Parameters:")
for key, value in BATES_PARAMS.items():
    print(f"  {key}: {value}")

# Price put option using put-call parity (Lewis)
print(f"\nPricing put option using Lewis (2001) via put-call parity...")
put_price_lewis = bates_model.put_price_put_call_parity(K, S0, r, T)

# Price put option using Monte Carlo
print(f"\nRunning Monte Carlo simulation with {MC_PATHS:,} paths...")
put_price_mc, std_error, ci_lower, ci_upper = MonteCarloUtils.price_put_option(
    bates_model, S0, r, T, K, MC_PATHS, MC_STEPS
)

print("\n" + "=" * 60)
print("PUT OPTION PRICING RESULTS")
print("=" * 60)
print(f"Strike Price: ${K:.2f} (95% of ${S0:.2f})")
print(f"Maturity: 70 days")
print(f"\nLewis Method (Put-Call Parity):")
print(f"  Put Price: ${put_price_lewis:.4f}")
print(f"\nMonte Carlo Method ({MC_PATHS:,} paths):")
print(f"  Put Price: ${put_price_mc:.4f}")
print(f"  95% Confidence Interval: [${ci_lower:.4f}, ${ci_upper:.4f}]")
print(f"  Standard Error: ${std_error:.6f}")
print(f"\nPrice Difference (Lewis - MC): ${put_price_lewis - put_price_mc:.4f}")

# Save results to file
with open(os.path.join(output_dir, 'put_option_price.txt'), 'w') as f:
    f.write("Put Option Pricing Results\n")
    f.write("=" * 40 + "\n")
    f.write(f"Product: European Put Option\n")
    f.write(f"Underlying: SM Energy Company\n")
    f.write(f"Current Price: ${S0:.2f}\n")
    f.write(f"Strike Price: ${K:.2f} ({MONEYNESS*100:.0f}% moneyness)\n")
    f.write(f"Maturity: 70 days\n")
    f.write(f"Risk-free Rate: {r*100:.2f}%\n\n")
    f.write(f"Bates Model Parameters:\n")
    for key, value in BATES_PARAMS.items():
        f.write(f"  {key}: {value}\n")
    f.write(f"\nPricing Results:\n")
    f.write(f"  Lewis Method: ${put_price_lewis:.4f}\n")
    f.write(f"  Monte Carlo Method: ${put_price_mc:.4f}\n")
    f.write(f"  95% Confidence Interval: [${ci_lower:.4f}, ${ci_upper:.4f}]\n")
    f.write(f"  Standard Error: ${std_error:.6f}\n")

# Create payoff distribution plot
print("\nGenerating payoff distribution...")
S_paths, _ = bates_model.simulate_paths(S0, r, T, 50000, MC_STEPS)
S_T = S_paths[:, -1]
payoffs = np.maximum(K - S_T, 0)

fig1, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram of final prices
axes[0].hist(S_T, bins=100, density=True, alpha=0.7, color='skyblue', edgecolor='black')
axes[0].axvline(x=K, color='red', linestyle='--', linewidth=2, label=f'Strike: ${K:.2f}')
axes[0].axvline(x=S0, color='green', linestyle='--', linewidth=2, label=f'Spot: ${S0:.2f}')
axes[0].set_xlabel('Stock Price at Maturity ($)')
axes[0].set_ylabel('Probability Density')
axes[0].set_title('Distribution of Final Stock Prices')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Histogram of payoffs
axes[1].hist(payoffs[payoffs > 0], bins=50, density=True, alpha=0.7, color='coral', edgecolor='black')
axes[1].axvline(x=put_price_mc, color='blue', linestyle='--', linewidth=2, label=f'Put Price: ${put_price_mc:.4f}')
axes[1].set_xlabel('Put Option Payoff ($)')
axes[1].set_ylabel('Probability Density')
axes[1].set_title('Put Option Payoff Distribution')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'payoff_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()

# Create price surface (strike vs maturity)
print("\nGenerating price surface...")
strikes_grid = np.linspace(S0 * 0.7, S0 * 1.3, 20)
maturities_grid = np.array([30, 50, 70, 90, 110]) / TRADING_DAYS

price_surface = np.zeros((len(maturities_grid), len(strikes_grid)))

for i, T_test in enumerate(maturities_grid):
    for j, K_test in enumerate(strikes_grid):
        price_surface[i, j] = bates_model.put_price_put_call_parity(K_test, S0, r, T_test)

fig2 = plt.figure(figsize=(12, 8))
ax2 = fig2.add_subplot(111, projection='3d')
X, Y = np.meshgrid(strikes_grid, maturities_grid * TRADING_DAYS)
ax2.plot_surface(X, Y, price_surface, cmap='viridis', alpha=0.8)
ax2.set_xlabel('Strike Price ($)')
ax2.set_ylabel('Maturity (Days)')
ax2.set_zlabel('Put Option Price ($)')
ax2.set_title('Put Option Price Surface (Bates Model)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'price_surface.png'), dpi=300, bbox_inches='tight')
plt.close()

# Convergence analysis
path_counts = [1000, 5000, 10000, 50000, 100000]
prices_mc, errors = MonteCarloUtils.convergence_analysis(
    bates_model, S0, r, T, K, path_counts, MC_STEPS, option_type='put'
)

convergence_df = pd.DataFrame({
    'Paths': path_counts,
    'Monte_Carlo_Price': prices_mc,
    'Lewis_Price': [put_price_lewis] * len(path_counts),
    'Std_Error': errors
})
convergence_df.to_csv(os.path.join(output_dir, 'simulation_results.csv'), index=False)

fig3 = VisualizationUtils.plot_convergence(path_counts, prices_mc, put_price_lewis)
fig3.suptitle('Put Option Price Convergence', fontsize=14)
VisualizationUtils.save_figure(fig3, os.path.join(output_dir, 'convergence_analysis.png'))

print("\n" + "=" * 60)
print("OUTPUT FILES GENERATED:")
print(f"1. {output_dir}/put_option_price.txt - Pricing results")
print(f"2. {output_dir}/payoff_distribution.png - Payoff distribution")
print(f"3. {output_dir}/price_surface.png - Price surface")
print(f"4. {output_dir}/convergence_analysis.png - Convergence plot")
print(f"5. {output_dir}/simulation_results.csv - Simulation data")
print("=" * 60)

print("\nPut option pricing completed successfully!")