import numpy as np
import matplotlib.pyplot as plt

H_file = "H.txt"
output_file = "IPR_results.txt"
plot_file = "IPR_vs_energy.png"

H = np.loadtxt(H_file)
eigvals, eigvecs = np.linalg.eigh(H)

IPR = np.sum(np.abs(eigvecs)**4, axis=0)
P = 1.0 / IPR

data = np.column_stack((eigvals, IPR, P))
np.savetxt(output_file, data, fmt="%.6e", header="eigval(eV)    IPR    ParticipationNumber(1/IPR)")

plt.figure(figsize=(6, 4))
plt.scatter(eigvals, IPR, s=12, c='dodgerblue', alpha=0.7)
plt.xlabel("Energy (eV)", fontsize=12)
plt.ylabel("IPR", fontsize=12)
plt.title("Inverse Participation Ratio vs Energy")
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig(plot_file, dpi=300)
plt.close()

print(f"Plot saved as {plot_file}")
