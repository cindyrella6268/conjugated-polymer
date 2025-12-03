import numpy as np

def write_lammps_dump(filename, coords, eigvecs, eigenvals=None, box=None, atom_type=1):
    N = coords.shape[0]
    M = eigvecs.shape[1]

    if box is None:
        padding = 5.0
        xmin, ymin, zmin = coords.min(axis=0) - padding
        xmax, ymax, zmax = coords.max(axis=0) + padding
        box = [xmin, xmax, ymin, ymax, zmin, zmax]

    with open(filename, 'w') as f:
        for mo_idx in range(M):
            v = eigvecs[:, mo_idx]
            psi2 = np.abs(v)**2         # per site psi2
            # normalize to [0,1] for color mapping
            # psi2_norm = (psi2 - psi2.min()) / (psi2.max() - psi2.min())

            f.write("ITEM: TIMESTEP\n")
            f.write(f"{mo_idx}\n")
            f.write("ITEM: NUMBER OF ATOMS\n")
            f.write(f"{N}\n")
            f.write("ITEM: BOX BOUNDS pp pp pp\n")
            f.write(f"{box[0]:.6f} {box[1]:.6f}\n")
            f.write(f"{box[2]:.6f} {box[3]:.6f}\n")
            f.write(f"{box[4]:.6f} {box[5]:.6f}\n")
            # add psi2 column
            f.write("ITEM: ATOMS id type x y z psi2\n")
            for i in range(N):
                x,y,z = coords[i]
                f.write(f"{i+1} {atom_type} {x:.6f} {y:.6f} {z:.6f} {psi2[i]:.6e}\n")

if __name__ == "__main__":
    N = 200
    H = np.loadtxt("H_onsite_11252025.txt")
    coords = np.loadtxt("coords.txt")
    # diagonalize
    evals, evecs = np.linalg.eigh(H)  # eigenvectors (200x200)
    order = np.argsort(evals)  # reorder into ascending
    evecs = evecs[:, order]

    write_lammps_dump("mo_dump.lammpstrj", coords, evecs)
    print("Wrote mo_dump.lammpstrj with", evecs.shape[1], "frames")
