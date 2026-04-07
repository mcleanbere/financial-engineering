"""
CIR Model Calibration
Full Team - Interest Rate Term Structure
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
print("CIR Model Calibration")
print("Euribor Term Structure Analysis")
print("=" * 80)

# Euribor term structure data
euribor_data = {
    'Tenor': ['1 week', '1 month', '3 months', '6 months', '12 months'],
    'Days': [7, 30, 90, 180, 365],
    'Rate': [0.00648, 0.00679, 0.01173, 0.01809, 0.02556]
}

euribor_df = pd.DataFrame(euribor_data)
euribor_df['T'] = euribor_df['Days'] / 365
euribor_df['Rate_Pct'] = euribor_df['Rate'] * 100

print("\nEuribor Term Structure Data:")
print(euribor_df[['Tenor', 'T', 'Rate_Pct']].to_string(index=False))

# Cubic spline interpolation
from scipy.interpolate import CubicSpline

T_points = euribor_df['T'].values
rate_points = euribor_df['Rate'].values

# Create weekly grid for 1 year (52 weeks)
weekly_T = np.arange(1/52, 1.01, 1/52)
cubic_spline = CubicSpline(T_points, rate_points, bc_type='natural')
weekly_rates = cubic_spline(weekly_T)

print(f"\nInterpolated Weekly Rates (52 weeks):")
print(f"  Min: {np.min(weekly_rates)*100:.4f}%")
print(f"  Max: {np.max(weekly_rates)*100:.4f}%")
print(f"  Mean: {np.mean(weekly_rates)*100:.4f}%")
print(f"  Std Dev: {np.std(weekly_rates)*100:.4f}%")

# Calibrate CIR model
r0 = euribor_df[euribor_df['Tenor'] == '1 week']['Rate'].values[0]
print(f"\nCurrent 1-week Euribor (r₀): {r0*100:.4f}%")

def calibrate_cir(tenors, yields, r0):
    """Calibrate CIR parameters to term structure"""
    from scipy.optimize import minimize, differential_evolution
    
    def objective(params):
        a, b, sigma = params
        
        if a <= 0 or b <= 0 or sigma <= 0:
            return 1e10
        
        model = CIRModel(a, b, sigma)
        model_yields = []
        
        for T in tenors:
            y = model.yield_to_maturity(r0, 0, T)
            model_yields.append(y)
        
        mse = np.mean((np.array(model_yields) - yields)**2)
        return mse
    
    bounds = [(0.01, 5), (0.001, 0.1), (0.01, 0.5)]
    result = minimize(objective, [0.5, 0.02, 0.1], method='L-BFGS-B', bounds=bounds)
    
    if result.success:
        return CIRModel(*result.x), result.fun
    else:
        result = differential_evolution(objective, bounds, maxiter=100)
        return CIRModel(*result.x), result.fun

print("\nCalibrating CIR model...")
cir_model, mse = calibrate_cir(T_points, rate_points, r0)

print("\n" + "=" * 60)
print("CIR CALIBRATION RESULTS")
print("=" * 60)
print(f"a (Mean reversion speed): {cir_model.a:.4f}")
print(f"b (Long-term mean): {cir_model.b:.6f} ({cir_model.b*100:.4f}%)")
print(f"σ (Volatility): {cir_model.sigma:.4f}")
print(f"Calibration MSE: {mse:.6f}")
print(f"Calibration RMSE: {np.sqrt(mse):.6f}")

# Save parameters to file
with open(os.path.join(output_dir, 'calibrated_params.txt'), 'w') as f:
    f.write("CIR Model Calibration Results\n")
    f.write("=" * 50 + "\n")
    f.write("Euribor Term Structure Calibration\n\n")
    f.write("Calibrated Parameters:\n")
    f.write(f"  a (Mean reversion speed): {cir_model.a:.4f}\n")
    f.write(f"  b (Long-term mean): {cir_model.b:.6f} ({cir_model.b*100:.4f}%)\n")
    f.write(f"  σ (Volatility): {cir_model.sigma:.4f}\n")
    f.write(f"\nCalibration MSE: {mse:.6f}\n")
    f.write(f"Calibration RMSE: {np.sqrt(mse):.6f}\n")

# Compute model yields for comparison
model_yields = []
for T in T_points:
    y = cir_model.yield_to_maturity(r0, 0, T)
    model_yields.append(y)

# Create comparison DataFrame
comparison_df = pd.DataFrame({
    'Tenor': euribor_df['Tenor'],
    'T_Years': T_points,
    'Market_Yield_Pct': rate_points * 100,
    'Model_Yield_Pct': np.array(model_yields) * 100,
    'Difference_Pct': (np.array(model_yields) - rate_points) * 100
})
comparison_df.to_csv(os.path.join(output_dir, 'calibration_fit.csv'), index=False)

print("\nYield Comparison:")
print(comparison_df.to_string(index=False))

# Create term structure plot
fig1 = plt.figure(figsize=(12, 5))

# Subplot 1: Term structure with interpolation
ax1 = plt.subplot(1, 2, 1)
ax1.plot(T_points, rate_points * 100, 'ro', markersize=8, label='Market Data')
ax1.plot(weekly_T, weekly_rates * 100, 'b-', linewidth=2, label='Cubic Spline')
ax1.plot(T_points, np.array(model_yields) * 100, 'g--', linewidth=2, marker='s', 
         markersize=6, label='CIR Model')
ax1.set_xlabel('Time to Maturity (Years)')
ax1.set_ylabel('Interest Rate (%)')
ax1.set_title('Euribor Term Structure: Market vs CIR Model')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Subplot 2: Calibration errors
ax2 = plt.subplot(1, 2, 2)
ax2.bar(range(len(T_points)), comparison_df['Difference_Pct'], 
        color=['green' if x > 0 else 'red' for x in comparison_df['Difference_Pct']])
ax2.set_xticks(range(len(T_points)))
ax2.set_xticklabels(euribor_df['Tenor'], rotation=45)
ax2.set_xlabel('Tenor')
ax2.set_ylabel('Model Error (basis points)')
ax2.set_title('CIR Model Calibration Errors')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'term_structure.png'), dpi=300, bbox_inches='tight')
plt.close()

# Create spline interpolation plot
fig2 = VisualizationUtils.plot_term_structure(
    T_points, rate_points * 100, weekly_T, weekly_rates * 100
)
fig2.suptitle('Euribor Term Structure - Cubic Spline Interpolation', fontsize=14)
VisualizationUtils.save_figure(fig2, os.path.join(output_dir, 'spline_interpolation.png'))

print("\n" + "=" * 60)
print("OUTPUT FILES GENERATED:")
print(f"1. {output_dir}/calibrated_params.txt - CIR parameters")
print(f"2. {output_dir}/term_structure.png - Term structure plot")
print(f"3. {output_dir}/spline_interpolation.png - Spline interpolation")
print(f"4. {output_dir}/calibration_fit.csv - Calibration comparison")
print("=" * 60)

print("\nCIR calibration completed successfully!")