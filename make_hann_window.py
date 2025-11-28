import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
N = 500
t = np.linspace(0, 1, N)
freq = 10  # Hz for the sine wave

# --- Signals ---
hann = np.hanning(N)
sine = np.sin(2 * np.pi * freq * t)
sine_windowed = sine * hann

plt.figure(figsize=(7, 3))
plt.plot(t, sine, label="Original sine")
plt.plot(t, sine_windowed, label="Windowed sine", linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend()
plt.tight_layout()

plt.show()
