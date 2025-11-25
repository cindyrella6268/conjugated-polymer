import numpy as np
from math import exp
import os

tphi_path = "tphi_values.txt"    # thru-bond coupling
coords_path = "coords_wrapped.txt"       # x y z
normals_path = "normals.txt"     # nx ny nz

n_chains = 20
n_monomers_per_chain = 10
N = n_chains * n_monomers_per_chain

# parameters (DOUBLE CHECK THESE)
J_inter_eV = 0.1
sigma = 5
r0 = 0.75 * sigma       # reference separation
alpha = 1.0 / sigma     # decay constant

# Through-space pair selection (cut off r DOUBLE CHECK)
through_space_cutoff = 15.0   # 3 sigma(?)
add_to_existing = True
exclude_bonded_pairs = True   # skip (i,j) that are bonded neighbors along same chain (they already have t(phi))

# Diagonal on-site energy (H_ii)
epsilon_default = 0.0   # from paper

def minimum_image(rij_vec, box_lengths):
    rij = np.array(rij_vec, dtype=float)
    L = np.array(box_lengths, dtype=float)
    return rij - np.rint(rij / L) * L

def load_tphi(tphi_path):
    # read tphi_values; one chain per line
    chains = []
    with open(tphi_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = [float(x) for x in line.split()]
            chains.append(vals)
    # Sanity check
    n_chains = len(chains)
    n_dihed = [len(c) for c in chains]
    if len(set(n_dihed)) != 1:
        raise ValueError("Inconsistent dihedral counts across chains: " + str(n_dihed))
    n_dihed_per_chain = n_dihed[0]
    n_mon_per_chain = n_dihed_per_chain + 1
    if n_chains * n_mon_per_chain != N:
        # allow mismatches but warn
        print("WARNING: file yields n_chains=%d, n_per_chain=%d -> total sites=%d (expected %d)"
              % (n_chains, n_mon_per_chain, n_chains * n_mon_per_chain, N))
    return chains

def global_index(chain_idx, monomer_idx, n_per_chain=n_monomers_per_chain):
    return int(chain_idx * n_per_chain + monomer_idx)

def load_onsite_energies(path):
    energies = np.zeros(N, dtype=float)
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            mon_id = int(parts[0])
            e_val = float(parts[1])
            energies[mon_id] = e_val
    return energies

def build_H_from_tphi(tphi_chains, epsilon=epsilon_default):
    # infer sizes
    n_chains_local = len(tphi_chains)
    n_dihed_per_chain = len(tphi_chains[0])
    n_per_chain_local = n_dihed_per_chain + 1
    N_local = n_chains_local * n_per_chain_local
    H = np.zeros((N_local, N_local), dtype=float)
    # diagonals
    np.fill_diagonal(H, epsilon)
    # fill through-bond couplings
    for c, chain_vals in enumerate(tphi_chains):
        for m, t in enumerate(chain_vals):   # m=0..n_dihed_per_chain-1
            i = global_index(c, m, n_per_chain_local)
            j = global_index(c, m+1, n_per_chain_local)
            H[i, j] = t
            H[j, i] = t
    return H

def load_coords_or_none(path):
    if not os.path.exists(path):
        return None
    if path.endswith(".npy"):
        return np.load(path)
    else:
        return np.loadtxt(path)

def normalize_vectors(v):
    v = np.array(v, dtype=float)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return v / norms

def compute_w_nm(fn, fm, rvec, J_inter=J_inter_eV, alpha=alpha, r0=r0):
    r = np.linalg.norm(rvec)
    if r == 0.0:
        return 0.0
    rhat = rvec / r        #new parameters?
    a = np.dot(fn, rhat)          #need to calculate distance with minimum image convention
    b = np.dot(fm, rhat)
    c = np.dot(fn, fm)
    # squares as in Eq (2)
    orient_factor = (a**2) * (b**2) * (c**2)
    expo = np.exp(-alpha * (r - r0))
    w = J_inter * orient_factor * expo
    return float(w)

def add_through_space_to_H(H, coords, normals, box_lengths,
                           cutoff=through_space_cutoff,
                           exclude_bonded=True,
                           tol_w = 1e-12):
    coords = np.asarray(coords, dtype=float)
    normals = np.asarray(normals, dtype=float)
    normals = normalize_vectors(normals)
    N_local = coords.shape[0]
    assert H.shape[0] == N_local, "H and coords length mismatch"

    # convenience helpers for chain/m index from global index
    def idx_to_chain_mon(idx):
        c = idx // n_monomers_per_chain
        m = idx % n_monomers_per_chain
        return int(c), int(m)

    pairs_added = 0
    total_w_added = 0.0
    for i in range(N_local):
        for j in range(i+1, N_local):
            # quick bbox by cutoff
            rij_vec = coords[j] - coords[i]
            rij_vec = minimum_image(rij_vec, box_lengths)
            r = np.linalg.norm(rij_vec)
            if r > cutoff:
                continue
            # exclude bonded neighbours along same chain
            ##changed to excluded intrachain interactions
            ci, mi = idx_to_chain_mon(i)
            cj, mj = idx_to_chain_mon(j)
            if ci == cj:
                continue
            # compute w
            w = compute_w_nm(normals[i], normals[j], rij_vec)
            if abs(w) > tol_w:
                if add_to_existing:
                    H[i, j] += w
                    H[j, i] += w
                else:
                    H[i, j] = w
                    H[j, i] = w
                pairs_added += 1
                total_w_added += 2*abs(w)   # symmetrically added
    return H, pairs_added, total_w_added

if __name__ == "__main__":
    # 1) load tphi
    tphi_chains = load_tphi(tphi_path)
    print("Loaded tphi for %d chains." % len(tphi_chains))

    # 2) build H with through-bond only
    H = build_H_from_tphi(tphi_chains, epsilon=epsilon_default)
    print("Built H with through-bond couplings. H.shape =", H.shape)
    # add on site energy diagonals
    onsite_energies = load_onsite_energies("onsite_backbone_eV.txt")
    np.fill_diagonal(H, onsite_energies)
    print("Updated Hamiltonian diagonal with on-site energies from LAMMPS.")

    # diagnostic: counts before through-space
    unique_offdiag = np.count_nonzero(np.triu(H, 1))
    print("Unique (upper-triangle) nonzero off-diagonal entries before w_nm:", unique_offdiag)

    # 3) try to load coords & normals
    coords = load_coords_or_none(coords_path)
    normals = load_coords_or_none(normals_path)

    if coords is None or normals is None:
        print("coords or normals not provided; skipping through-space addition.")
        if coords is None:
            print(" - coords file not found at:", coords_path)
        if normals is None:
            print(" - normals file not found at:", normals_path)
    else:
        #define box length
        xlo, xhi = -12.80611686042665, 12.80611686042665
        ylo, yhi = -20.09222334179953, 20.09222334179953
        zlo, zhi = -20.290272100505682, 20.290272100505682
        box_lengths = np.array([xhi - xlo, yhi - ylo, zhi - zlo])

        # 4) compute and add through-space couplings
        H, pairs_added, total_w = add_through_space_to_H(H, coords, normals, box_lengths, cutoff=through_space_cutoff)
        print(f"Through-space couplings added for {pairs_added} pairs (symmetric additions), total amplitude sum ~ {total_w:.6f} eV")

        # diagnostic after adding through-space
        unique_offdiag_after = np.count_nonzero(np.triu(H, 1))
        print("Unique off-diagonal nonzero entries after w_nm:", unique_offdiag_after)
        print("New entries added (upper triangle):", unique_offdiag_after - unique_offdiag)

    # 5) Save H to disk (optional)
    #np.save("H_200x200_with_w.npy", H)
    np.savetxt("H_onsite_backbone.txt", H, fmt="%.12f")
    #print("Saved H_200x200_with_w.npy and .txt")
