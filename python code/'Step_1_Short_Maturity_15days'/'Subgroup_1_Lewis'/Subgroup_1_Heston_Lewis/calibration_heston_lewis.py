"""
Sub-group 1: Heston Model Calibration via Lewis (2001)
Members 1-3
15-day maturity options
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

#pathverification
print("Root directory added to sys.path:", root_dir)

# Import custom modules
from Shared_Modules.heston_model import HestonModel
from Shared_Modules.calibration_utils import calibration_utils
from Shared_Modules.visualization_utils import VisualizationUtils

# Constants
S0 = 232.90
r = 0.015
TRADING_DAYS = 250
T = 15 / TRADING_DAYS  # 15 days to years

# Create output directory
output_dir = 'Outputs'
os.makedirs(output_dir, exist_ok=True)

print("=" * 80)
print("SUB-GROUP 1: Heston Model Calibration via Lewis (2001)")
print("Maturity: 15 days")
print("=" * 80)

# Load market data (replace with actual Excel file)
def load_market_data():
    """Load option data from Excel file"""
    # Path to Excel file
    excel_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'Data', 'option_data.xlsx'
    )
    
    try:
        df = pd.read_excel(excel_path)
        # Filter for 15-day maturity options
        df['days_to_maturity'] = df.get('maturity_days', 15)
        df_filtered = df[abs(df['days_to_maturity'] - 15) <= 2].copy()
        
        # Use put-call parity for put options
        if 'put_price' in df_filtered.columns and 'call_price' not in df_filtered.columns:
            df_filtered['call_price'] = df_filtered['put_price'] + \
                                        df_filtered['strike'] * np.exp(-r * T) - S0
        
        return df_filtered[['strike', 'call_price']].dropna()
    
    except FileNotFoundError:
        print("Warning: Option data file not found. Using synthetic data for demonstration.")
        # Generate synthetic data for demonstration
        true_model = HestonModel(2.45, 0.043, 0.58, -0.73, 0.041)
        strikes = np.linspace(S0 * 0.8, S0 * 1.2, 15)
        data = []
        for K in strikes:
            price = true_model.call_price_lewis(K, S0, r, T)
            data.append({'strike': K, 'call_price': price})
        return pd.DataFrame(data)

# Load data
market_data = load_market_data()
print(f"\nLoaded {len(market_data)} options")

# Calibrate Heston model
print("\nCalibrating Heston model using Lewis (2001)...")
calibrator = CalibrationUtils()
heston_model, mse = calibrator.calibrate_heston_lewis(market_data, S0, r, T)

# Display results
print("\n" + "=" * 60)
print("CALIBRATION RESULTS")
print("=" * 60)
params = heston_model.get_parameters()
for key, value in params.items():
    print(f"{key}: {value:.6f}")
print(f"MSE: {mse:.6f}")
print(f"RMSE: {np.sqrt(mse):.6f}")

# Save parameters to file
with open(os.path.join(output_dir, 'calibrated_params.txt'), 'w') as f:
    f.write("Heston Model Calibration Results (Lewis 2001)\n")
    f.write("=" * 50 + "\n")
    f.write(f"Maturity: 15 days\n")
    f.write(f"Risk-free rate: {r*100:.2f}%\n")
    f.write(f"Spot price: ${S0:.2f}\n\n")
    f.write("Calibrated Parameters:\n")
    for key, value in params.items():
        f.write(f"{key}: {value:.6f}\n")
    f.write(f"\nMSE: {mse:.6f}\n")
    f.write(f"RMSE: {np.sqrt(mse):.6f}\n")

# Compute model prices for all strikes
strikes = market_data['strike'].values
market_prices = market_data['call_price'].values
model_prices = [heston_model.call_price_lewis(K, S0, r, T) for K in strikes]

# Create comparison DataFrame
comparison_df = pd.DataFrame({
    'Strike': strikes,
    'Market_Price': market_prices,
    'Model_Price': model_prices,
    'Difference': model_prices - market_prices,
    'Abs_Error': np.abs(model_prices - market_prices)
})
comparison_df.to_csv(os.path.join(output_dir, 'model_vs_market_prices.csv'), index=False)

print(f"\nAverage Absolute Error: ${comparison_df['Abs_Error'].mean():.4f}")
print(f"Max Absolute Error: ${comparison_df['Abs_Error'].max():.4f}")

# Create visualization
fig = VisualizationUtils.plot_calibration_fit(
    strikes, market_prices, model_prices, S0,
    'Heston Model Calibration (Lewis 2001) - 15-day Maturity'
)
VisualizationUtils.save_figure(fig, os.path.join(output_dir, 'calibration_fit.png'))

# Create parameter comparison with typical ranges
fig2, ax2 = plt.subplots(figsize=(10, 6))
param_names = list(params.keys())
param_values = list(params.values())
colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in param_values]
ax2.bar(param_names, param_values, color=colors, alpha=0.7)
ax2.set_xlabel('Parameter', fontsize=12)
ax2.set_ylabel('Value', fontsize=12)
ax2.set_title('Calibrated Heston Parameters', fontsize=14)
ax2.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'parameter_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

print("\n" + "=" * 60)
print("OUTPUT FILES GENERATED:")
print(f"1. {output_dir}/calibrated_params.txt - Parameter values")
print(f"2. {output_dir}/calibration_fit.png - Market vs model prices")
print(f"3. {output_dir}/parameter_comparison.png - Parameter visualization")
print(f"4. {output_dir}/model_vs_market_prices.csv - Price comparison")
print("=" * 60)

print("\nCalibration completed successfully!")