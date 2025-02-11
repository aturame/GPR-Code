import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, ExpSineSquared
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.fft import fft, fftfreq
from joblib import Parallel, delayed
import time

# Load Data
openprice = pd.read_excel("/Users/atulramesh/Desktop/data2.xlsx", usecols=['PX_OPEN'])
date = pd.read_excel("/Users/atulramesh/Desktop/data2.xlsx", usecols=['Dates'])

# Date Processing
date_numeric = pd.to_datetime(date['Dates']).astype('int64') // 10**9
date_numeric = date_numeric.values.reshape(-1, 1)

# Scaling Dates and Prices
date_scaler = MinMaxScaler()
date_numeric_scaled = date_scaler.fit_transform(date_numeric)

price_scaler = StandardScaler()
openprice_scaled = price_scaler.fit_transform(openprice.values.reshape(-1, 1)).flatten()

# Kernel Setup
kernel_rbf = C(1.0, (1e-6, 1e6)) * RBF(length_scale=15.0, length_scale_bounds=(1e-1, 1e3))
kernel_periodic = C(1.0, (1e-6, 1e6)) * ExpSineSquared(length_scale=10.0, periodicity=30.0)
Kernel_Combined = kernel_rbf + kernel_periodic

# Parallelized GPR Fitting
def parallel_fit(seed):
    gpr = GaussianProcessRegressor(
        kernel=Kernel_Combined,
        n_restarts_optimizer=0,  # No internal restarts as we handle them with Joblib
        alpha=1e-2,
        random_state=seed,
        optimizer ='fmin_l_bfgs_b'
    )
    gpr.fit(date_numeric_scaled, openprice_scaled)
    return gpr

# Timing the Process
start_time = time.perf_counter()

# Parallel Execution with Joblib
n_restarts = 5  # Number of parallel optimizations
models = Parallel(n_jobs=-1)(delayed(parallel_fit)(seed) for seed in range(n_restarts))

# Select the best model based on log-marginal likelihood
best_model = max(models, key=lambda m: m.log_marginal_likelihood())

# Predictions
prices_pred_scaled, sigma = best_model.predict(date_numeric_scaled, return_std=True)
prices_pred = price_scaler.inverse_transform(prices_pred_scaled.reshape(-1, 1)).flatten()

# FFT Analysis on the Predicted GPR Curve
N = len(prices_pred)
T = (date_numeric_scaled[1] - date_numeric_scaled[0])[0]
fft_values = fft(prices_pred)
fft_frequencies = fftfreq(N, T)[:N // 2]
fft_magnitude = 2.0 / N * np.abs(fft_values[:N // 2])

# Dominant Frequency
if len(fft_magnitude) > 0:
    dominant_freq_index = np.argmax(fft_magnitude)
    dominant_frequency = fft_frequencies[dominant_freq_index]
    dominant_amplitude = fft_magnitude[dominant_freq_index]

    print(f"Dominant Frequency: {dominant_frequency} Hz")
    print(f"Dominant Amplitude: {dominant_amplitude}")

# End Timer
end_time = time.perf_counter()
print(f"Processing Time (Parallel): {end_time - start_time:.4f} seconds")
# Plotting GPR Prediction
dates_plot = pd.to_datetime(date['Dates'])
plt.figure(figsize=(12, 6))
plt.scatter(dates_plot, openprice, color="red", label="NG1:COM DATA", alpha=0.6)
plt.plot(dates_plot, prices_pred, label="Fitted Curve (RBF+Periodic)", color="blue")
plt.fill_between(dates_plot, prices_pred - sigma, prices_pred + sigma, color="blue", alpha=0.2)
plt.title("Natural Gas Prices Prediction using GPR")
plt.xlabel("Dates")
plt.ylabel("Price")
plt.legend()

# Plotting FFT Spectrum
plt.figure(figsize=(12, 6))
plt.plot(fft_frequencies, fft_magnitude, color='purple')
plt.title("Frequency Spectrum (FFT of GPR Prediction)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()
# End Timer
