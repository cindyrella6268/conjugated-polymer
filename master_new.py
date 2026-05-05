import numpy as np
import csv
from typing import List, Tuple
import os
import matplotlib.pyplot as plt
import time
from collections import Counter

from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import connected_components

# System parameters
J_inter_eV = 0.1
sigma = 5.0
r0 = 0.75 * sigma
alpha = 1.0 / sigma
through_space_cutoff = 15.0

kB = 8.617333262145e-5    # eV / K
hbar = 6.582119569e-16    # eV·s
angstrom_to_m = 1e-10     # m per Å

lambda1 = 0.45            # eV  (reorganization energy)
T = 300.0                 # K
G = 0.005                 # dimensionless prefactor
LAM_MIN = 1e-8            # avoid division by zero

Ndop = 1                  # number of holes
F_mag = 20000             # V/cm

# Onsite energy file
onsite_file = "onsite_energies_all.txt"

# Polynomial coefficients for t(φ)
C = [-0.01, 1.275, 0.016, -0.870, -0.029, 0.540]

# Field vectors for tensor averaging
fields = [np.array([F_mag, 0.0, 0.0]),
          np.array([0.0, F_mag, 0.0]),
          np.array([0.0, 0.0, F_mag])]
field_labels = ["x", "y", "z"]

# Parsing
def parse_cg_dump(dump_file):
    with open(dump_file, "r") as f:
        lines = f.readlines()

    frames = []
    current_frame = []
    current_timestep = None
    reading_atoms = False
    box_lengths = None

    for i, line in enumerate(lines):

        if line.startswith("ITEM: TIMESTEP"):
            if current_frame:
                frames.append((current_timestep, box_lengths, current_frame))
                current_frame = []
            current_timestep = int(lines[i + 1].strip())

        elif line.startswith("ITEM: BOX BOUNDS"):
            xlo, xhi = map(float, lines[i+1].split()[:2])
            ylo, yhi = map(float, lines[i+2].split()[:2])
            zlo, zhi = map(float, lines[i+3].split()[:2])
            box_lengths = np.array([xhi - xlo, yhi - ylo, zhi - zlo])

        elif line.startswith("ITEM: ATOMS"):
            reading_atoms = True

        elif line.startswith("ITEM:"):
            reading_atoms = False

        elif reading_atoms:
            parts = line.strip().split()
            if len(parts) < 11:
                continue
            atom_id = int(parts[0])
            # parts[1] is atom type — not mol_id, so we skip it
            x, y, z = map(float, parts[2:5])
            nx, ny, nz = map(float, parts[5:8])
            ix, iy, iz = map(int, parts[8:11])
            current_frame.append(
                (atom_id,
                 np.array([x, y, z]),
                 np.array([nx, ny, nz]),
                 ix, iy, iz)
            )

    if current_frame:
        frames.append((current_timestep, box_lengths, current_frame))

    return frames


# Chain assignment via ASE

def assign_mol_ids_from_geometry(coords, box_lengths, bond_cutoff=5.0):
    N = len(coords)
    graph = lil_matrix((N, N), dtype=np.int8)

    for i in range(N):
        for j in range(i + 1, N):
            rij = coords[j] - coords[i]
            rij = minimum_image(rij, box_lengths)
            if np.linalg.norm(rij) < bond_cutoff:
                graph[i, j] = 1
                graph[j, i] = 1

    n_chains, labels = connected_components(
        graph.tocsr(), directed=False)

    print(f"  Detected {n_chains} chains.")
    size_dist = Counter(Counter(labels).values())
    print(f"  Chain size distribution (size: count): "
          f"{ dict(sorted(size_dist.items())) }")

    return labels

def extract_coords_normals(frame):
    coords  = []
    normals = []
    for (_, coord, normal, _, _, _) in frame:
        coords.append(coord)
        normals.append(normal)
    return np.array(coords), np.array(normals)


def align_normals_consistently(normals):
    aligned = [normals[0]]
    for i in range(1, len(normals)):
        if np.dot(aligned[-1], normals[i]) < 0:
            aligned.append(-normals[i])
        else:
            aligned.append(normals[i])
    return aligned


def minimum_image(rij_vec, box_lengths):
    return rij_vec - np.rint(rij_vec / box_lengths) * box_lengths


def normalize_vectors(v):
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return v / norms


# Transfer integrals along chains  (variable-length, mol-id based)
def compute_tphi_chains(frame, atom_mol_ids, C=C):
    mol_normals       = {}
    mol_to_global_idx = {}

    for global_idx, (atom_id, coord, normal, ix, iy, iz) in enumerate(frame):
        mol_id = int(atom_mol_ids[global_idx])
        if mol_id not in mol_normals:
            mol_normals[mol_id]       = []
            mol_to_global_idx[mol_id] = []
        mol_normals[mol_id].append(normal)
        mol_to_global_idx[mol_id].append(global_idx)

    tphi_chains = {}
    for mol_id, normals in mol_normals.items():
        normals = align_normals_consistently(normals)
        tphi_chains[mol_id] = []
        for i in range(len(normals) - 1):
            cos_theta = np.clip(np.dot(normals[i], normals[i + 1]), -1.0, 1.0)
            angle_deg = np.degrees(np.arccos(cos_theta))
            tphi_chains[mol_id].append(t_phi_from_angle(angle_deg, C))

    return tphi_chains, mol_to_global_idx


def t_phi_from_angle(angle_deg, C):
    phi_rad = np.radians(angle_deg)
    cos_phi = np.cos(phi_rad)
    return sum(c * (cos_phi ** i) for i, c in enumerate(C))


# Hamiltonian construction
def build_H_from_tphi(tphi_chains, mol_to_global_idx, N_total):
    H = np.zeros((N_total, N_total))

    for mol_id, t_vals in tphi_chains.items():
        atom_indices = mol_to_global_idx[mol_id]
        for bond_i, t in enumerate(t_vals):
            i = atom_indices[bond_i]
            j = atom_indices[bond_i + 1]
            H[i, j] = t
            H[j, i] = t

    return H


def add_through_space_to_H(H, coords, normals, box_lengths, atom_mol_ids):
    normals = normalize_vectors(normals)
    N_local = coords.shape[0]
    pairs_added = 0

    for i in range(N_local):
        for j in range(i + 1, N_local):
            if atom_mol_ids[i] == atom_mol_ids[j]:
                continue
            rij_vec = coords[j] - coords[i]
            rij_vec = minimum_image(rij_vec, box_lengths)
            r = np.linalg.norm(rij_vec)
            if r > through_space_cutoff:
                continue
            w = compute_w_nm(normals[i], normals[j], rij_vec)
            if np.abs(w) > 1e-12:
                H[i, j] += w
                H[j, i] += w
                pairs_added += 1

    return H, pairs_added


def compute_w_nm(fn, fm, rvec):
    fn   = np.array(fn).flatten()
    fm   = np.array(fm).flatten()
    rvec = np.array(rvec).flatten()
    r    = np.linalg.norm(rvec)
    if r == 0:
        return 0
    rhat = rvec / r
    a = np.dot(fn, rhat)
    b = np.dot(fm, rhat)
    c = np.dot(fn, fm)
    orient_factor = (a * a) * (b * b) * (c * c)
    expo = np.exp(-alpha * (r - r0))
    return J_inter_eV * orient_factor * expo


# Onsite energies
def load_onsite_energies_from_file(filename, N):
    onsite_dict = {}
    with open(filename, "r") as f:
        for line in f:
            if line.strip().startswith("#") or len(line.strip()) == 0:
                continue
            parts    = line.split()
            timestep = int(parts[0])
            monomer  = int(parts[1]) - 1   # convert to 0-based index
            energy   = float(parts[2])
            if timestep not in onsite_dict:
                onsite_dict[timestep] = np.zeros(N)
            onsite_dict[timestep][monomer] = energy
    return onsite_dict


# Rate matrix  (Marcus / MO basis)
def get_MO_center_MIC(c2_a, coords, box_lengths):
    i_ref  = np.argmax(c2_a)
    r_shift = coords - coords[i_ref]
    r_shift = np.array([minimum_image(r, box_lengths) for r in r_shift])
    R_rel  = np.sum(c2_a[:, None] * r_shift, axis=0)
    R_abs  = R_rel + coords[i_ref]
    return R_abs


def build_kij(H, coords, box_lengths, F_vec):
    N = H.shape[0]
    eigvals, eigvecs = np.linalg.eigh(H)

    c2 = np.abs(eigvecs) ** 2          # (N_sites, N_MOs)

    # MIC-corrected MO centres
    R_mo = np.zeros((N, 3))
    for a in range(N):
        R_mo[a] = get_MO_center_MIC(c2[:, a], coords, box_lengths)

    # Reorganization energy matrix
    c4      = np.abs(eigvecs) ** 4
    sum_c4  = np.sum(c4, axis=0)
    lam     = lambda1 * (sum_c4.reshape((N, 1)) + sum_c4.reshape((1, N)))
    np.fill_diagonal(lam, 0)
    lam     = np.maximum(lam, LAM_MIN)

    # Effective electronic coupling V² in MO basis
    H_off   = H.copy()
    np.fill_diagonal(H_off, 0)
    V2      = H_off ** 2
    V2_mo   = G ** 2 * (c2.T @ V2 @ c2)
    V2_mo   = 0.5 * (V2_mo + V2_mo.T)

    # Field
    F_mag_local = np.linalg.norm(F_vec)
    F_hat       = F_vec / F_mag_local
    #F_eV_A      = F_mag_local * 1e-10   # V/cm → eV/Å
    F_eV_A      = F_mag_local * 1e-8

    prefactor = 2.0 * np.pi / hbar
    kij       = np.zeros((N, N))

    for a in range(N):
        for b in range(N):
            if a == b:
                continue
            if V2_mo[a, b] <= 0:
                continue
            lam_ab = lam[a, b]
            rij    = minimum_image(R_mo[b] - R_mo[a], box_lengths)
            #deltaG = (eigvals[b] - eigvals[a]) + F_eV_A * np.dot(rij, F_hat)
            deltaG = (eigvals[b] - eigvals[a]) - F_eV_A * np.dot(rij, F_hat)
            expo   = -((deltaG + lam_ab) ** 2) / (4.0 * lam_ab * kB * T)
            rate_pref = prefactor * V2_mo[a, b] / np.sqrt(4.0 * np.pi * lam_ab * kB * T)
            kij[a, b] = rate_pref * np.exp(expo)

    return kij, eigvals, eigvecs, R_mo, F_hat, F_mag_local

# Steady-state hole populations
def solve_hole_populations(kij, Ndop=Ndop, tol=1e-8, max_iter=100000,
                           damping=0.5, verbose=False):
    kij = np.array(kij, dtype=float)
    np.fill_diagonal(kij, 0.0)
    N = kij.shape[0]

    P  = np.full(N, Ndop / N, dtype=float)
    KT = kij.T

    for it in range(1, max_iter + 1):
        P_old = P.copy()
        for i in range(N):
            R_i = kij[i].sum()
            if R_i <= 0:
                continue
            num_i = np.dot(KT[i], P)
            s_i   = np.dot(kij[i] - KT[i], P)
            P_raw = (num_i / R_i) * (1.0 - s_i / R_i)
            if not np.isfinite(P_raw) or P_raw < 0:
                P_raw = 0.0
            P[i]  = damping * P_raw + (1.0 - damping) * P[i]

        P       *= Ndop / P.sum()
        residual = np.max(np.abs(P - P_old))
        if verbose and (it % 1000 == 0 or residual < tol):
            print(f"    Pop iter {it:6d}: residual = {residual:.3e}")
        if residual < tol:
            break

    return P

# Mobility
def compute_mobility(kij, R_mo, P, F_hat, F_mag_local, box_lengths, Ndop=Ndop):
    J_vec = np.zeros(3, dtype=float)
    N     = len(P)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if kij[i, j] <= 0.0:
                continue
            rij    = minimum_image(R_mo[j] - R_mo[i], box_lengths)
            rij_m  = rij * angstrom_to_m
            J_vec += kij[i, j] * P[i] * (1.0 - P[j]) * rij_m

    F_mag_SI = F_mag_local * 1e2        # V/cm → V/m
    mu_col   = J_vec / (F_mag_SI * Ndop)
    return mu_col, J_vec


def run_single_field(H, coords, box_lengths, F_vec, Ndop=Ndop, verbose=False):
    kij, eigvals, eigvecs, R_mo, F_hat, F_mag_local = build_kij(
        H, coords, box_lengths, F_vec)
    P = solve_hole_populations(kij, Ndop=Ndop, verbose=verbose)
    mu_col, J_vec = compute_mobility(
        kij, R_mo, P, F_hat, F_mag_local, box_lengths, Ndop=Ndop)
    return mu_col, J_vec


def compute_full_mobility_tensor(H, coords, box_lengths, Ndop=Ndop, verbose=False):
    mu_cols = []
    for F_vec, label in zip(fields, field_labels):
        mu_col, J_vec = run_single_field(
            H, coords, box_lengths, F_vec, Ndop=Ndop, verbose=verbose)
        mu_cols.append(mu_col)

    mu_full    = np.column_stack(mu_cols)       # 3×3 tensor in m²/V·s
    mu_full_cm = mu_full * 1e4                  # convert to cm²/V·s
    mu_eff     = np.trace(mu_full_cm) / 3.0     # isotropic effective mobility

    mu_x = mu_full_cm[0, 0]
    mu_y = mu_full_cm[1, 1]
    mu_z = mu_full_cm[2, 2]

    return mu_full_cm, mu_eff, mu_x, mu_y, mu_z


# Main loop
mobility_outfile = "mobility_results.txt"
with open(mobility_outfile, "w") as f:
    f.write("# timestep   time_ns      mu_x(cm2/Vs)   mu_y(cm2/Vs)   "
            "mu_z(cm2/Vs)   mu_eff(cm2/Vs)\n")
print(f"Mobility results will be written to: {mobility_outfile}")

dump_file = "cg_beads.dump"
frames    = parse_cg_dump(dump_file)
print(f"Total frames parsed from dump: {len(frames)}")

# N is determined from the first frame
N = len(frames[0][2])
print(f"Total monomers (from first frame): {N}")

onsite_dict = load_onsite_energies_from_file(onsite_file, N)
print(f"Loaded onsite energies for {len(onsite_dict)} timesteps: "
      f"{sorted(onsite_dict.keys())}")

frames = [(ts, box, frm) for ts, box, frm in frames if ts in onsite_dict]
print(f"Frames to process (matching onsite energies): {len(frames)}")

eff_mobility = []
time_frame   = []

for timestep, box_lengths, frame in frames:
    print(f"\n{'='*60}")
    print(f"Processing timestep: {timestep}")
    print(f"{'='*60}")

    # Sort atoms by atom_id for consistent ordering
    frame_sorted = sorted(frame, key=lambda x: x[0])
    coords, normals = extract_coords_normals(frame_sorted)

    # Assign chain IDs geometrically using ASE
    # Adjust bond_cutoff if chain detection looks wrong
    atom_mol_ids = assign_mol_ids_from_geometry(
        coords, box_lengths, bond_cutoff=5.0)

    # Report chain composition for this frame
    chain_sizes = Counter(atom_mol_ids)
    n_chains    = len(chain_sizes)
    print(f"  Chains: {n_chains}  |  sizes: "
          f"{ {k: v for k, v in sorted(chain_sizes.items())} }")

    # Intrachain transfer integrals
    tphi_chains, mol_to_global_idx = compute_tphi_chains(
        frame_sorted, atom_mol_ids)

    # Check onsite energies exist for this timestep
    if timestep not in onsite_dict:
        raise ValueError(f"No onsite energies found for timestep {timestep}")
    onsite_energies = onsite_dict[timestep]

    # Build Hamiltonian
    N_frame = len(frame_sorted)
    H = build_H_from_tphi(tphi_chains, mol_to_global_idx, N_frame)
    H, pairs_added = add_through_space_to_H(
        H, coords, normals, box_lengths, atom_mol_ids)
    np.fill_diagonal(H, onsite_energies)
    print(f"  Through-space pairs added: {pairs_added}")

    # Full 3×3 mobility tensor
    mu_full_cm, mu_eff, mu_x, mu_y, mu_z = compute_full_mobility_tensor(
        H, coords, box_lengths, Ndop=Ndop, verbose=False
    )

    time_ns_current = timestep * 0.1
    print(f"  μ_x   = {mu_x:.6f} cm²/V·s")
    print(f"  μ_y   = {mu_y:.6f} cm²/V·s")
    print(f"  μ_z   = {mu_z:.6f} cm²/V·s")
    print(f"  μ_eff = {mu_eff:.6f} cm²/V·s")

    eigvals_mu, eigvecs_mu = np.linalg.eig(mu_full_cm)
    idx_sort   = np.argsort(eigvals_mu)[::-1]
    eigvals_mu = eigvals_mu[idx_sort]
    print(f"  Principal mobilities: {eigvals_mu}")

    with open(mobility_outfile, "a") as f:
        f.write(f"{timestep:>10d}   {time_ns_current:>8.3f}   "
                f"{mu_x:>14.8f}   {mu_y:>14.8f}   "
                f"{mu_z:>14.8f}   {mu_eff:>14.8f}\n")

    eff_mobility.append(mu_eff)
    time_frame.append(timestep)

# Plot
eff_mobility = np.array(eff_mobility)
time_frame   = np.array(time_frame)
time_ns      = time_frame * 0.1

mean_val = np.mean(eff_mobility)
std_val  = np.std(eff_mobility)
sem      = std_val / np.sqrt(len(eff_mobility))

plt.figure(figsize=(6, 4))
sc = plt.scatter(time_ns, eff_mobility, c=eff_mobility, s=50,
                 alpha=0.7, cmap='viridis')
plt.errorbar(time_ns, eff_mobility, yerr=sem, fmt='none', capsize=3)
plt.axhline(0.0101, linestyle='--', color='r', label='Experiment = 0.0101')
plt.axhline(mean_val, linestyle='-', color='blue',
            label=f'Mean = {mean_val:.4f}')
plt.xlabel('Time (ns)')
plt.ylabel('Mobility (cm²/V·s)')
plt.legend()
plt.colorbar(sc, label='Mobility (cm²/V·s)')
plt.tight_layout()
plt.savefig("mobility_vs_time.png", dpi=300)
plt.close()

print(f"\nFinal: mean μ = {mean_val:.6f} ± {sem:.6f} cm²/V·s "
      f"over {len(eff_mobility)} frames")
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        465,13        All
