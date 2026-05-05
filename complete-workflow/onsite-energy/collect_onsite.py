import os
import numpy as np

RESULTS_DIR = "results"
N_FRAMES    = 31
N_MONOMERS  = 200

energy_matrix = np.full((N_FRAMES, N_MONOMERS), np.nan)
n_missing     = 0

with open("onsite_energies_all.txt", "w") as out:
    out.write("# frame_index\tmonomer_index\tE_onsite_eV\n")

    for fi in range(N_FRAMES):
        fpath = os.path.join(RESULTS_DIR,
                             f"onsite_energies_frame_{fi:03d}.txt")
        if not os.path.exists(fpath):
            print(f"[WARNING] Missing: {fpath}")
            n_missing += 1
            continue

        with open(fpath, "r") as f:
            for line in f:
                if line.startswith("#") or line.strip() == "":
                    continue
                parts = line.strip().split()
                fi_val = int(parts[0])
                mi_val = int(parts[1])    # 1-based
                e_val  = float(parts[2])
                energy_matrix[fi_val, mi_val - 1] = e_val
                out.write(f"{fi_val}\t{mi_val}\t{e_val:.10f}\n")

        print(f"  Collected frame {fi:03d}")

np.save("onsite_energies_all.npy", energy_matrix)

nan_count = np.isnan(energy_matrix).sum()
print(f"\nSaved onsite_energies_all.txt")
print(f"Saved onsite_energies_all.npy  shape={energy_matrix.shape}")
print(f"Missing frame files: {n_missing}  |  NaN entries: {nan_count}")
