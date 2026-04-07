"""
Sub-group 2: Heston Model Calibration via Carr-Madan (1999)
Members 4-6
15-day maturity options
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Shared_Modules.heston_model import HestonModel
from Shared_Modules.calibration_utils import CalibrationUtils
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
print("SUB-GROUP 2: Heston Model Calibration via Carr-Madan (1999)")
print("Maturity: 15 days")
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
        df['days_to_maturity'] = df.get('maturity_days', 15)
        df_filtered = df[abs(df['days_to_maturity'] - 15) <= 2].copy()
        
        if 'put_price' in df_filtered.columns and 'call_price' not in df_filtered.columns:
            df_filtered['call_price'] = df_filtered['put_price'] + \
                                        df_filtered['strike'] * np.exp(-r * T) - S0
        
        return df_filtered[['strike', 'call_price']].dropna()
    
    except FileNotFoundError:
        print("Warning: Option data file not found. Using synthetic data.")
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

# Calibrate Heston model using Carr-Madan
print("\nCalibrating Heston model using Carr-Madan (1999) FFT...")
calibrator = CalibrationUtils()
heston_model_cm, mse_cm = calibrator.calibrate_heston_carrmadan(market_data, S0, r, T)

# Also calibrate using Lewis for comparison
heston_model_lewis, mse_lewis = calibrator.calibrate_heston_lewis(market_data, S0, r, T)

# Display results
print("\n" + "=" * 60)
print("CALIBRATION RESULTS COMPARISON")
print("=" * 60)

print("\nCarr-Madan (FFT) Results:")
params_cm = heston_model_cm.get_parameters()
for key, value in params_cm.items():
    print(f"{key}: {value:.6f}")
print(f"MSE: {mse_cm:.6f}")

print("\nLewis (Integration) Results:")
params_lewis = heston_model_lewis.get_parameters()
for key, value in params_lewis.items():
    print(f"{key}: {value:.6f}")
print(f"MSE: {mse_lewis:.6f}")

print("\nParameter Differences:")
for key in params_cm.keys():
    diff = params_cm[key] - params_lewis[key]
    print(f"{key}: {diff:+.6f}")

# Save parameters to file
with open(os.path.join(output_dir, 'calibrated_params.txt'), 'w') as f:
    f.write("Heston Model Calibration Results - Comparison\n")
    f.write("=" * 50 + "\n")
    f.write(f"Maturity: 15 days\n\n")
    
    f.write("Carr-Madan (FFT) Method:\n")
    for key, value in params_cm.items():
        f.write(f"  {key}: {value:.6f}\n")
    f.write(f"  MSE: {mse_cm:.6f}\n\n")
    
    f.write("Lewis (Integration) Method:\n")
    for key, value in params_lewis.items():
        f.write(f"  {key}: {value:.6f}\n")
    f.write(f"  MSE: {mse_lewis:.6f}\n\n")
    
    f.write("Differences:\n")
    for key in params_cm.keys():
        f.write(f"  {key}: {params_cm[key] - params_lewis[key]:+.6f}\n")

# Compute FFT prices
def carr_madan_prices_quick(model, strikes, alpha=1.0, N=4096):
    """Quick FFT pricing"""
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
    
    from scipy.fft import fft
    fft_vals = fft(psi * weights) * eta / 3
    call_prices = np.exp(-alpha * k) / np.pi * np.real(fft_vals)
    
    results = {}
    for K in strikes:
        k_val = np.log(K / S0)
        idx = np.argmin(np.abs(k - k_val))
        results[K] = max(call_prices[idx], 1e-6)
    
    return results

# Compute prices for comparison
strikes = market_data['strike'].values
market_prices = market_data['call_price'].values

# Lewis prices
lewis_prices = [heston_model_lewis.call_price_lewis(K, S0, r, T) for K in strikes]

# Carr-Madan prices
cm_prices_dict = carr_madan_prices_quick(heston_model_cm, strikes)
cm_prices = [cm_prices_dict[K] for K in strikes]

# Create comparison DataFrame
comparison_df = pd.DataFrame({
    'Strike': strikes,
    'Market_Price': market_prices,
    'Lewis_Price': lewis_prices,
    'CarrMadan_Price': cm_prices,
    'Lewis_Error': np.abs(lewis_prices - market_prices),
    'CarrMadan_Error': np.abs(cm_prices - market_prices)
})
comparison_df.to_csv(os.path.join(output_dir, 'fft_prices.csv'), index=False)

print(f"\nAverage Absolute Error - Lewis: ${comparison_df['Lewis_Error'].mean():.4f}")
print(f"Average Absolute Error - Carr-Madan: ${comparison_df['CarrMadan_Error'].mean():.4f}")

# Create comparison plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Model fits
axes[0].scatter(strikes, market_prices, color='red', s=50, alpha=0.7, label='Market')
axes[0].plot(strikes, lewis_prices, 'b-', linewidth=2, label='Lewis (2001)')
axes[0].plot(strikes, cm_prices, 'g--', linewidth=2, label='Carr-Madan (1999)')
axes[0].axvline(x=S0, color='black', linestyle=':', label=f'Spot: ${S0}')
axes[0].set_xlabel('Strike Price ($)')
axes[0].set_ylabel('Option Price ($)')
axes[0].set_title('Model Comparison: Lewis vs Carr-Madan')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Differences
axes[1].plot(strikes, comparison_df['Lewis_Error'], 'b-', linewidth=2, label='Lewis Error')
axes[1].plot(strikes, comparison_df['CarrMadan_Error'], 'g--', linewidth=2, label='Carr-Madan Error')
axes[1].set_xlabel('Strike Price ($)')
axes[1].set_ylabel('Absolute Pricing Error ($)')
axes[1].set_title('Pricing Error Comparison')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'lewis_vs_carrmadan.png'), dpi=300, bbox_inches='tight')
plt.close()

# Create calibration fit plot for Carr-Madan
fig2 = VisualizationUtils.plot_calibration_fit(
    strikes, market_prices, cm_prices, S0,
    'Carr-Madan Calibration - 15-day Maturity'
)
VisualizationUtils.save_figure(fig2, os.path.join(output_dir, 'calibration_fit.png'))

print("\n" + "=" * 60)
print("OUTPUT FILES GENERATED:")
print(f"1. {output_dir}/calibrated_params.txt - Parameter values")
print(f"2. {output_dir}/calibration_fit.png - Carr-Madan fit")
print(f"3. {output_dir}/lewis_vs_carrmadan.png - Method comparison")
print(f"4. {output_dir}/fft_prices.csv - Price comparison")
print("=" * 60)

print("\nCalibration completed successfully!")