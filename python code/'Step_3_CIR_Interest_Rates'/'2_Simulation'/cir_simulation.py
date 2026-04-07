"""
CIR Model Monte Carlo Simulation
Full Team - 12-month Euribor Forecasting
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Shared_Modules.cir_model import CIRModel
from Shared_Modules.visualization_utils import VisualizationUtils

# Create output directory
output_dir = 'Outputs'
os.makedirs(output_dir, exist_ok=True)

print("=" * 80)
print("CIR Model Monte Carlo Simulation")
print("12-month Euribor Forecasting")
print("=" * 80)

# CIR parameters from calibration
CIR_PARAMS = {
    'a': 0.87,
    'b': 0.0245,
    'sigma': 0.11
}

# Initial rate (1-week Euribor)
r0 = 0.00648

# Simulation parameters
T = 1.0  # 1 year forecast
N_PATHS = 100000
N_STEPS = 365  # Daily steps

print(f"\nSimulation Parameters:")
print(f"  Initial Rate (1-week Euribor): {r0*100:.4f}%")
print(f"  Forecast Horizon: {T} year (12 months)")
print(f"  Number of Paths: {N_PATHS:,}")
print(f"  Time Steps: {N_STEPS} (daily)")
print(f"\nCIR Model Parameters:")
for key, value in CIR_PARAMS.items():
    print(f"  {key}: {value}")

# Initialize CIR model
cir_model = CIRModel(**CIR_PARAMS)

# Run simulation
print("\nRunning Monte Carlo simulation...")
simulation_results = cir_model.simulate_future_rate(r0, T, N_PATHS, N_STEPS)

final_rates = simulation_results['final_rates']
mean_rate = simulation_results['mean']
std_rate = simulation_results['std']
ci_lower = simulation_results['percentile_2.5']
ci_upper = simulation_results['percentile_97.5']

print("\n" + "=" * 60)
print("SIMULATION RESULTS")
print("=" * 60)
print(f"Expected 12-month Euribor in 1 year: {mean_rate*100:.4f}%")
print(f"Standard Deviation: {std_rate*100:.4f}%")
print(f"95% Confidence Interval: [{ci_lower*100:.4f}%, {ci_upper*100:.4f}%]")
print(f"Median Rate: {np.median(final_rates)*100:.4f}%")
print(f"Min Rate: {np.min(final_rates)*100:.4f}%")
print(f"Max Rate: {np.max(final_rates)*100:.4f}%")

# Calculate additional statistics
prob_above_3pct = np.mean(final_rates > 0.03) * 100
prob_above_4pct = np.mean(final_rates > 0.04) * 100
prob_below_2pct = np.mean(final_rates < 0.02) * 100

print(f"\nProbabilities:")
print(f"  Rate > 3%: {prob_above_3pct:.2f}%")
print(f"  Rate > 4%: {prob_above_4pct:.2f}%")
print(f"  Rate < 2%: {prob_below_2pct:.2f}%")

# Save results to file
with open(os.path.join(output_dir, 'simulation_results.txt'), 'w') as f:
    f.write("CIR Model Monte Carlo Simulation Results\n")
    f.write("=" * 50 + "\n")
    f.write(f"Initial Rate (1-week Euribor): {r0*100:.4f}%\n")
    f.write(f"Forecast Horizon: 12 months\n")
    f.write(f"Number of Paths: {N_PATHS:,}\n\n")
    f.write("CIR Model Parameters:\n")
    for key, value in CIR_PARAMS.items():
        f.write(f"  {key}: {value}\n")
    f.write("\nSimulation Results:\n")
    f.write(f"  Expected 12-month Euribor: {mean_rate*100:.4f}%\n")
    f.write(f"  Standard Deviation: {std_rate*100:.4f}%\n")
    f.write(f"  95% Confidence Interval: [{ci_lower*100:.4f}%, {ci_upper*100:.4f}%]\n")
    f.write(f"  Median Rate: {np.median(final_rates)*100:.4f}%\n")
    f.write(f"  Min Rate: {np.min(final_rates)*100:.4f}%\n")
    f.write(f"  Max Rate: {np.max(final_rates)*100:.4f}%\n\n")
    f.write("Probability Analysis:\n")
    f.write(f"  Rate > 3%: {prob_above_3pct:.2f}%\n")
    f.write(f"  Rate > 4%: {prob_above_4pct:.2f}%\n")
    f.write(f"  Rate < 2%: {prob_below_2pct:.2f}%\n")

# Create distribution plot
fig1 = VisualizationUtils.plot_rate_distribution(
    final_rates, mean_rate, ci_lower, ci_upper,
    'Simulated 12-month Euribor Distribution (1-year Horizon)'
)
VisualizationUtils.save_figure(fig1, os.path.join(output_dir, 'rate_distribution.png'))

# Create sample paths plot
sample_paths = simulation_results['paths'][:50, :]  # Take 50 sample paths
fig2 = VisualizationUtils.plot_sample_paths(sample_paths, n_samples=50)
fig2.suptitle('Sample CIR Interest Rate Paths (50 Simulations)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'sample_paths.png'), dpi=300, bbox_inches='tight')
plt.close()

# Create quantile plot
fig3, ax3 = plt.subplots(figsize=(12, 6))

time_grid = np.linspace(0, T, N_STEPS + 1)
quantiles = [2.5, 25, 50, 75, 97.5]
colors = ['#d62728', '#ff7f0e', '#2ca02c', '#ff7f0e', '#d62728']
labels = ['2.5th Percentile', '25th Percentile', 'Median', '75th Percentile', '97.5th Percentile']

for q, color, label in zip(quantiles, colors, labels):
    q_vals = np.percentile(simulation_results['paths'], q, axis=0)
    ax3.plot(time_grid * 365, q_vals * 100, color=color, linewidth=2, label=label)

ax3.fill_between(time_grid * 365, 
                  np.percentile(simulation_results['paths'], 2.5, axis=0) * 100,
                  np.percentile(simulation_results['paths'], 97.5, axis=0) * 100,
                  alpha=0.2, color='gray')
ax3.set_xlabel('Time (Days)')
ax3.set_ylabel('Interest Rate (%)')
ax3.set_title('CIR Model Forecast: Euribor 12-month Rate Evolution')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'rate_forecast_paths.png'), dpi=300, bbox_inches='tight')
plt.close()

# Create confidence intervals table
confidence_levels = [50, 75, 90, 95, 99]
ci_table = []
for cl in confidence_levels:
    lower = np.percentile(final_rates, (100 - cl) / 2)
    upper = np.percentile(final_rates, 100 - (100 - cl) / 2)
    ci_table.append({
        'Confidence_Level': f'{cl}%',
        'Lower_Bound_Pct': lower * 100,
        'Upper_Bound_Pct': upper * 100,
        'Range_Pct': (upper - lower) * 100
    })

ci_df = pd.DataFrame(ci_table)
ci_df.to_csv(os.path.join(output_dir, 'confidence_intervals.csv'), index=False)

print("\nConfidence Intervals:")
print(ci_df.to_string(index=False))

print("\n" + "=" * 60)
print("OUTPUT FILES GENERATED:")
print(f"1. {output_dir}/simulation_results.txt - Simulation statistics")
print(f"2. {output_dir}/rate_distribution.png - Rate distribution")
print(f"3. {output_dir}/sample_paths.png - Sample paths")
print(f"4. {output_dir}/rate_forecast_paths.png - Forecast paths with quantiles")
print(f"5. {output_dir}/confidence_intervals.csv - Confidence intervals")
print("=" * 60)

print("\nCIR simulation completed successfully!")