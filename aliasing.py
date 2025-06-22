import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import erf

# -----------------------------
# Seaborn / Matplotlib global styling
# -----------------------------
sns.set_theme(
    style="whitegrid",        # white background with grid
    palette="deep",           # deep color palette
    font="sans-serif",
    rc={
        "figure.figsize": (10, 8),
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "lines.linewidth": 2
    }
)

# -----------------------------
# Parameters
# -----------------------------
K = 20                # original bandlimit (max Fourier mode)
M = 512               # number of time samples (should be >> 2*K)
L = 2 * np.pi         # period length
dx = L / M
x = np.linspace(-np.pi, np.pi - dx, M)  # sample points in [-π, π)

# -----------------------------
# Construct a real, band-limited function f(x)
# f(x) = sum_{k=1..K} [c_k cos(kx) + s_k sin(kx)]
# with random Gaussian coefficients
# -----------------------------
np.random.seed(0)
c = np.random.randn(K) / np.sqrt(K)
s = np.random.randn(K) / np.sqrt(K)
f = np.zeros_like(x)
for k in range(K):
    f += c[k] * np.cos(k * x) + s[k] * np.sin(k * x)

# -----------------------------
# Define activation functions
# -----------------------------
def relu(u):
    return np.maximum(0, u)

def gelu(u):
    # Gaussian Error Linear Unit
    return 0.5 * u * (1 + erf(u / np.sqrt(2)))

def sigmoid(u):
    return 1 / (1 + np.exp(-u))

def tanh_act(u):
    return np.tanh(u)

# Apply activations pointwise
f_relu = relu(f)
f_gelu = gelu(f)
f_sig  = sigmoid(f)
f_tanh = tanh_act(f)

# -----------------------------
# Compute shifted FFT magnitude spectrum
# -----------------------------
def spectrum(u):
    U = np.fft.fft(u)
    U = np.fft.fftshift(U) / M   # shift zero freq to center & normalize
    return np.abs(U)

spec_f    = spectrum(f)
spec_relu = spectrum(f_relu)
spec_gelu = spectrum(f_gelu)
spec_sig  = spectrum(f_sig)
spec_tanh = spectrum(f_tanh)

# Frequency axis: integer Fourier modes from -M/2 to M/2-1
k = np.arange(-M//2, M//2)

# -----------------------------
# Plot time-domain and frequency-domain
# -----------------------------
fig, (ax1, ax2) = plt.subplots(2, 1)

# Time-domain signals
ax1.plot(x, f,     label='f (band-limited)')
ax1.plot(x, f_relu, label='ReLU(f)')
ax1.plot(x, f_gelu, label='GELU(f)')
ax1.plot(x, f_sig,  label='sigmoid(f)')
ax1.plot(x, f_tanh, label='tanh(f)')
ax1.set_xlim(-np.pi, np.pi)
ax1.set_ylabel('Amplitude')
ax1.set_title('Time Domain')
ax1.legend(loc='upper right')

# Frequency-domain magnitudes
ax2.plot(k, spec_f,    label='|F|')
ax2.plot(k, spec_relu, label='|ReLU(F)|')
ax2.plot(k, spec_gelu, label='|GELU(F)|')
ax2.plot(k, spec_sig,  label='|sigmoid(F)|')
ax2.plot(k, spec_tanh, label='|tanh(F)|')
# Mark the Nyquist cutoff at ±K
ax2.axvline(+K, color='red', linestyle='--', label='Nyquist ±K')
ax2.axvline(-K, color='red', linestyle='--')
ax2.set_xlim(-60, 60)
ax2.set_xlabel('Fourier mode k')
ax2.set_ylabel('Magnitude')
ax2.set_title('Frequency Domain')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig('figs/aliasing_effects.pdf', dpi=300)
plt.show()
