import numpy as np
from scipy.optimize import bisect

# Load energies
energies = np.loadtxt("energies.txt")

# Parameters
kB = 8.617333262e-5  # eV/K
T = 300              # K
n = 0.05             # carrier concentration (fraction of occupied sites)

def total_occupancy(EF):
    p = 1.0 / (np.exp((energies - EF) / (kB * T)) + 1.0)
    return np.sum(p) - len(energies) * n

# Find EF so that total_occupancy(EF) = 0
EF = bisect(total_occupancy, -1.0, 1.0)

# Compute occupations
p0 = 1.0 / (np.exp((energies - EF) / (kB * T)) + 1.0)

np.savetxt("P0.txt", p0, fmt="%.6f")

print(f"Fermi level EF = {EF:.4f} eV")
print("Saved initial populations p0.txt.")
