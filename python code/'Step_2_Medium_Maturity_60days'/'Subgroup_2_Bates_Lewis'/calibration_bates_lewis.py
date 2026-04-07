"""
Sub-group 2: Bates Model Calibration via Lewis (2001)
Members 4-6
60-day maturity options
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Shared_Modules.bates_model import BatesModel
from Shared_Modules.calibration_utils import CalibrationUtils
from Shared_Modules.visualization_utils import VisualizationUtils

# Constants
S0 = 232.90
r = 0.015
TRADING_DAYS = 250
T = 60 / TRADING_DAYS  # 60 days to years

# Create output directory
output_dir = 'Outputs'
os.makedirs(output_dir, exist_ok=True)

print("=" * 80)
print("SUB-GROUP 2: Bates Model Calibration via Lewis (2001)")
print("Maturity: 60 days")
print("=" * 80)

# Load market data
def load_market_data():
    """Load option data from Excel file"""
    excel_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'Data', 'option_data.xlsx'
    )
    
    try:
        df = pd.read_excel(excel_path)
        df['days_to_maturity'] = df.get('maturity_days', 60)
        df_filtered = df[abs(df['days_to_maturity'] - 60) <= 5].copy()
        
        if 'put_price' in df_filtered.columns and 'call_price' not in df_filtered.columns:
            df_filtered['call_price'] = df_filtered['put_price'] + \
                                        df_filtered['strike'] * np.exp(-r * T) - S0
        
        return df_filtered[['strike', 'call_price']].dropna()
    
    except FileNotFoundError:
        print("Warning: Option data file not found. Using synthetic data.")
        true_model = BatesModel(2.81, 0.047, 0.54, -0.71, 0.044, 0.23, -0.018, 0.15)
        strikes = np.linspace(S0 * 0.7, S0 * 1.3, 20)
        data = []
        for K in strikes:
            price = true_model.call_price_lewis(K, S0, r, T)
            data.append({'strike': K, 'call_price': price})
        return pd.DataFrame(data)

# Load data
market_data = load_market_data()
print(f"\nLoaded {len(market_data)} options")

# Calibrate Bates model using Lewis
print("\nCalibrating Bates model using Lewis (2001) integration...")

def objective_bates(params):
    """MSE objective function for Bates calibration"""
    kappa, theta, sigma, rho, v0, lambd, mu_j, sigma_j = params
    
    # Parameter bounds check
    if (kappa <= 0 or kappa > 20 or theta <= 0 or theta > 1 or
        sigma <= 0 or sigma > 3 or abs(rho) >= 1 or v0 <= 0 or v0 > 1 or
        lambd < 0 or lambd > 5 or abs(mu_j) > 0.5 or sigma_j <= 0 or sigma_j > 1):
        return 1e10
    
    model = BatesModel(kappa, theta, sigma, rho, v0, lambd, mu_j, sigma_j)
    
    mse = 0
    for _, row in market_data.iterrows():
        model_price = model.call_price_lewis(row['strike'], S0, r, T)
        mse += (model_price - row['call_price'])**2
    
    return mse / len(market_data)

from scipy.optimize import minimize, differential_evolution

# Initial parameters
initial_params = [2.5, 0.045, 0.55, -0.72, 0.042, 0.2, -0.02, 0.15]
bounds = [(0.01, 20), (0.001, 1), (0.01, 3), (-0.99, 0.99), (0.001, 1),
          (0, 5), (-0.5, 0.5), (0.01, 1)]

print("Optimizing...")
result = minimize(objective_bates, initial_params, method='L-BFGS-B', bounds=bounds)

if result.success:
    params = result.x
    mse = result.fun
else:
    print("Local optimization failed, trying differential evolution...")
    result = differential_evolution(objective_bates, bounds, maxiter=100, popsize=20)
    params = result.x
    mse = result.fun

bates_model = BatesModel(*params)

# Display results
print("\n" + "=" * 60)
print("CALIBRATION RESULTS - Bates Model (Lewis 2001)")
print("=" * 60)

param_names = ['kappa', 'theta', 'sigma', 'rho', 'v0', 'lambd', 'mu_j', 'sigma_j']
for name, value in zip(param_names, params):
    print(f"{name}: {value:.6f}")
print(f"MSE: {mse:.6f}")
print(f"RMSE: {np.sqrt(mse):.6f}")

# Save parameters to file
with open(os.path.join(output_dir, 'calibrated_params.txt'), 'w') as f:
    f.write("Bates Model Calibration Results (Lewis 2001)\n")
    f.write("=" * 50 + "\n")
    f.write(f"Maturity: 60 days\n")
    f.write(f"Risk-free rate: {r*100:.2f}%\n")
    f.write(f"Spot price: ${S0:.2f}\n\n")
    f.write("Calibrated Parameters:\n")
    for name, value in zip(param_names, params):
        f.write(f"{name}: {value:.6f}\n")
    f.write(f"\nMSE: {mse:.6f}\n")
    f.write(f"RMSE: {np.sqrt(mse):.6f}\n")

# Compute model prices
strikes = market_data['strike'].values
market_prices = market_data['call_price'].values
model_prices = [bates_model.call_price_lewis(K, S0, r, T) for K in strikes]

# Create comparison DataFrame
comparison_df = pd.DataFrame({
    'Strike': strikes,
    'Market_Price': market_prices,
    'Model_Price': model_prices,
    'Difference': model_prices - market_prices,
    'Abs_Error': np.abs(model_prices - market_prices)
})
comparison_df.to_csv(os.path.join(output_dir, 'results_summary.csv'), index=False)

print(f"\nAverage Absolute Error: ${comparison_df['Abs_Error'].mean():.4f}")
print(f"Max Absolute Error: ${comparison_df['Abs_Error'].max():.4f}")

# Create calibration fit plot
fig1 = VisualizationUtils.plot_calibration_fit(
    strikes, market_prices, model_prices, S0,
    'Bates Model Calibration (Lewis 2001) - 60-day Maturity'
)
VisualizationUtils.save_figure(fig1, os.path.join(output_dir, 'calibration_fit.png'))

# Parameter sensitivity analysis
print("\nPerforming parameter sensitivity analysis...")

def parameter_sensitivity(param_idx, param_name, base_value, variation=0.2):
    """Analyze sensitivity of MSE to parameter changes"""
    variations = np.linspace(base_value * (1 - variation), base_value * (1 + variation), 20)
    mse_values = []
    
    for v in variations:
        test_params = list(params)
        test_params[param_idx] = v
        test_params = tuple(test_params)
        mse_values.append(objective_bates(test_params))
    
    return variations, mse_values

fig2, axes = plt.subplots(2, 4, figsize=(16, 10))
axes = axes.flatten()

sensitive_params = ['kappa', 'theta', 'sigma', 'rho', 'v0', 'lambd', 'mu_j', 'sigma_j']

for i, (name, value) in enumerate(zip(sensitive_params, params)):
    variations, mse_values = parameter_sensitivity(i, name, value)
    axes[i].plot(variations, mse_values, 'b-', linewidth=2)
    axes[i].axvline(x=value, color='r', linestyle='--', linewidth=2, label='Optimal')
    axes[i].set_xlabel(f'{name}')
    axes[i].set_ylabel('MSE')
    axes[i].set_title(f'Sensitivity: {name}')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'parameter_sensitivity.png'), dpi=300, bbox_inches='tight')
plt.close()

print("\n" + "=" * 60)
print("OUTPUT FILES GENERATED:")
print(f"1. {output_dir}/calibrated_params.txt - Parameter values")
print(f"2. {output_dir}/calibration_fit.png - Calibration fit plot")
print(f"3. {output_dir}/parameter_sensitivity.png - Sensitivity analysis")
print(f"4. {output_dir}/results_summary.csv - Price comparison")
print("=" * 60)

print("\nBates model calibration completed successfully!")