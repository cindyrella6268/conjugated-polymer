import numpy as np
import csv
from typing import List, Tuple
import os
import matplotlib.pyplot as plt
import time

# system parameters
n_chains = 20
n_monomers_per_chain = 10
N = n_chains * n_monomers_per_chain          # 200 total monomers

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

# onsite energy file
onsite_file = "onsite_energies_all.txt"

# Polynomial coefficients for t(φ)
C = [-0.01, 1.275, 0.016, -0.870, -0.029, 0.540]

# Field vectors for tensor averaging
fields = [np.array([F_mag, 0, 0]),
          np.array([0, F_mag, 0]),
          np.array([0, 0, F_mag])]
field_labels = ["x", "y", "z"]

# Parse LAMMPS dump
# input: cg_beads.dump file path (string)
# output: list of (timestep, box_lengths, h, h_inv, atom_list)
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
                frames.append((current_timestep, box_lengths, h, h_inv, current_frame))
                current_frame = []
            current_timestep = int(lines[i + 1].strip())

        elif line.startswith("ITEM: BOX BOUNDS"):
            xlo, xhi, xy = map(float, lines[i+1].split())
            ylo, yhi, xz = map(float, lines[i+2].split())
            zlo, zhi, yz = map(float, lines[i+3].split())
            Lx = xhi - xlo
            Ly = yhi - ylo
            Lz = zhi - zlo
            box_lengths = np.array([Lx, Ly, Lz])
            h = np.array([[Lx, xy, xz],
                          [0.0, Ly, yz],
                          [0.0, 0.0, Lz]])
            h_inv = np.linalg.inv(h)

        elif line.startswith("ITEM: ATOMS"):
            reading_atoms = True
        elif line.startswith("ITEM:"):
            reading_atoms = False
        elif reading_atoms:
            parts = line.strip().split()
            if len(parts) < 11:
                continue
            atom_id = int(parts[0])
            mol_type = int(parts[1])
            x, y, z = map(float, parts[2:5])
            nx, ny, nz = map(float, parts[5:8])
            ix, iy, iz = map(int, parts[8:11])
            current_frame.append(
                (atom_id, mol_type,
                 np.array([x, y, z]),
                 np.array([nx, ny, nz]),
                 ix, iy, iz)
            )

    if current_frame:
        frames.append((current_timestep, box_lengths, h, h_inv, current_frame))

    return frames

# Write frame as LAMMPS data file
# innput: sorted atom list, template .data file, output path
# output: writes frame_{timestep}.data to disk
def write_frame_as_data(frame, template_data, outfile):
    with open(template_data, "r") as f:
        lines = f.readlines()

    coords = {atom_id: coord for (atom_id, _, coord, _, _, _, _) in frame}
    new_lines = []
    in_atoms = False

    for line in lines:
        if line.startswith("Atoms"):
            in_atoms = True
            new_lines.append(line)
            continue
        if in_atoms and line.strip() == "":
            new_lines.append(line)
            continue
        if in_atoms:
            parts = line.split()
            if len(parts) < 7:
                new_lines.append(line)
                continue
            atom_id = int(parts[0])
            if atom_id in coords:
                x, y, z = coords[atom_id]
                parts[4] = f"{x}"
                parts[5] = f"{y}"
                parts[6] = f"{z}"
                line = " ".join(parts) + "\n"
        new_lines.append(line)

    with open(outfile, "w") as f:
        f.writelines(new_lines)

# Extract coords and normals from frame
# input: sorted atom list
# output: coords (N×3, Å), normals (N×3, unit vectors)
def extract_coords_normals(frame):
    coords = []
    normals = []
    for (_, _, coord, normal, _, _, _) in frame:
        coords.append(coord)
        normals.append(normal)
    return np.array(coords), np.array(normals)

# Align normals within a chain to be consistently oriented
def align_normals_consistently(normals):
    aligned = [normals[0]]
    for i in range(1, len(normals)):
        if np.dot(aligned[-1], normals[i]) < 0:
            aligned.append(-normals[i])
        else:
            aligned.append(normals[i])
    return aligned

# Compute inter-normal dihedral angles along each chain
# input: sorted atom list
# output: list of (chain_id, monomer_index, angle_deg)
def compute_dihedrals_for_frame(frame, n_chains=n_chains, beads_per_chain=n_monomers_per_chain):
    chains = [[] for _ in range(n_chains)]
    for idx, (_, _, _, normal, _, _, _) in enumerate(frame):
        chain_id = idx // beads_per_chain
        chains[chain_id].append(normal)

    dihedrals = []
    for chain_id, normals in enumerate(chains):
        normals = align_normals_consistently(normals)
        for i in range(len(normals) - 1):
            cos_theta = np.clip(np.dot(normals[i], normals[i+1]), -1.0, 1.0)
            angle_deg = np.degrees(np.arccos(cos_theta))
            dihedrals.append((chain_id, i, angle_deg))

    return dihedrals

# t(φ): intrachain transfer integral from dihedral angle
# input: angle in degrees, polynomial coefficients C
# output: scalar transfer integral (eV)
def t_phi_from_angle(angle_deg, C):
    phi_rad = np.radians(angle_deg)
    cos_phi = np.cos(phi_rad)
    return sum(c * (cos_phi**i) for i, c in enumerate(C))

def compute_tphi(dihedrals, C=C):
    angles = [angle for (_, _, angle) in dihedrals]
    return np.array([t_phi_from_angle(a, C) for a in angles])

# input: flat tphi array (length n_chains × (beads-1))
# output: list of lists, one per chain, each of length beads-1
def reshape_tphi_into_chains(tphi, n_chains=n_chains, beads_per_chain=n_monomers_per_chain):
    dihed_per_chain = beads_per_chain - 1
    return [list(tphi[c*dihed_per_chain:(c+1)*dihed_per_chain])
            for c in range(n_chains)]

# Load onsite energies for all timesteps
# input: onsite energy txt file
# output: np.array of length n_monomer
def load_onsite_energies_from_file(filename, N):
    onsite_dict = {}

    with open(filename, "r") as f:
        for line in f:
            if line.strip().startswith("#") or len(line.strip()) == 0:
                continue

            parts = line.split()
            timestep = int(parts[0])
            monomer = int(parts[1]) - 1   # convert to 0-based index
            energy = float(parts[2])

            if timestep not in onsite_dict:
                onsite_dict[timestep] = np.zeros(N)

            onsite_dict[timestep][monomer] = energy

    return onsite_dict

# Build intrachain Hamiltonian
# input: tphi_chains (list of lists of transfer integrals)
# output: N×N matrix H with intrachain off-diagonals only
#         (diagonal is zero here; onsite energies added later)
def build_H_from_tphi(tphi_chains):             # REMOVED epsilon parameter
    n_per_chain = len(tphi_chains[0]) + 1
    N_local = len(tphi_chains) * n_per_chain
    H = np.zeros((N_local, N_local))
    # diagonal left as zero — onsite energies assigned after SLURM results

    for c, chain_vals in enumerate(tphi_chains):
        for m, t in enumerate(chain_vals):
            i = c * n_per_chain + m
            j = c * n_per_chain + m + 1
            H[i, j] = t
            H[j, i] = t

    return H

# normalize rows of a matrix
def normalize_vectors(v):
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return v / norms

# Through-space interchain coupling w_nm
# input: normal vectors fn, fm; displacement vector rvec (Å)
# output: scalar coupling w_nm (eV)
def compute_w_nm(fn, fm, rvec):
    fn = np.array(fn).flatten()
    fm = np.array(fm).flatten()
    rvec = np.array(rvec).flatten()
    r = np.linalg.norm(rvec)
    if r == 0:
        return 0
    rhat = rvec / r
    a = np.dot(fn, rhat)
    b = np.dot(fm, rhat)
    c = np.dot(fn, fm)
    orient_factor = (a*a) * (b*b) * (c*c)
    expo = np.exp(-alpha * (r - r0))
    return J_inter_eV * orient_factor * expo

# MIC for triclinic box
def minimum_image_triclinic(rij_vec, h, h_inv):
    rij_vec = np.array(rij_vec).flatten()
    s = h_inv @ rij_vec
    s -= np.rint(s)
    return (h @ s).flatten()

# add interchain through-space couplings to H
# INPUT:  H (N×N), coords, normals, box info
# OUTPUT: H updated with W_nm for all interchain pairs within cutoff
def add_through_space_to_H(H, coords, normals, box_lengths, h, h_inv):
    normals = normalize_vectors(normals)
    N_local = coords.shape[0]
    pairs_added = 0

    for i in range(N_local):
        for j in range(i+1, N_local):
            rij_vec = coords[j] - coords[i]
            rij_vec = minimum_image_triclinic(rij_vec, h, h_inv)
            r = np.linalg.norm(rij_vec)
            if r > through_space_cutoff:
                continue
            if i // n_monomers_per_chain == j // n_monomers_per_chain:
                continue   # skip intrachain pairs
            w = compute_w_nm(normals[i], normals[j], rij_vec)
            if np.abs(w) > 1e-12:
                H[i, j] += w
                H[j, i] += w
                pairs_added += 1

    return H, pairs_added

# Build Marcus rate matrix k_ij
# input: H (N×N full), coords, box info, field vector F_vec (V/cm)
# output: k_ij (N×N, s⁻¹), eigvals, eigvecs, F_hat
def build_kij(H, coords, box_lengths, h, h_inv, F_vec):
    N = H.shape[0]
    eigvals, eigvecs = np.linalg.eigh(H)
    c2 = np.abs(eigvecs)**2
    c4 = np.abs(eigvecs)**4
    sum_c4 = np.sum(c4, axis=0)
    lam = lambda1 * (sum_c4.reshape((N,1)) + sum_c4.reshape((1,N)))
    np.fill_diagonal(lam, 0)
    lam = np.maximum(lam, LAM_MIN)

    H_off = H.copy()
    np.fill_diagonal(H_off, 0)
    V2 = H_off**2
    temp = np.tensordot(c2, V2, axes=(0, 0))
    V2_mo = G**2 * (temp @ c2)
    V2_mo = 0.5 * (V2_mo + V2_mo.T)

    F_hat = F_vec / np.linalg.norm(F_vec)
    F_eV_A = np.linalg.norm(F_vec) * 1e-8   # V/cm → eV/Å
    pref = 2 * np.pi / hbar
    kij = np.zeros((N, N))

    for a in range(N):
        for b in range(N):
            if a == b:
                continue
            if V2_mo[a, b] <= 0:
                continue
            rij = minimum_image_triclinic(coords[b] - coords[a], h, h_inv)
            deltaG = (eigvals[b] - eigvals[a]) + F_eV_A * np.dot(rij, F_hat)
            expo = -((deltaG + lam[a,b])**2) / (4 * lam[a,b] * kB * T)
            rate_pref = pref * V2_mo[a,b] / np.sqrt(4 * np.pi * lam[a,b] * kB * T)
            kij[a, b] = rate_pref * np.exp(expo)

    return kij, eigvals, eigvecs, F_hat

# Solve steady-state hole populations
# input: k_ij (N×N), Ndop (number of holes)
# output: P (N,) occupation probabilities summing to Ndop
def solve_hole_populations(kij, Ndop=Ndop, tol=1e-8, max_iter=100000):
    kij = np.array(kij)
    np.fill_diagonal(kij, 0)
    N = kij.shape[0]
    P = np.full(N, Ndop / N)
    KT = kij.T

    for _ in range(max_iter):
        P_old = P.copy()
        in_flux = KT @ P               # VECTORIZED: was inner for-loop
        out_rate = kij.sum(axis=1)
        mask = out_rate > 0
        P[mask] = in_flux[mask] / out_rate[mask]
        P *= Ndop / P.sum()
        if np.max(np.abs(P - P_old)) < tol:
            break

    return P

# Compute mobility along one field direction
# input: k_ij, coords, box info, P, F_vec (V/cm)
# output: scalar mobility (m²/V·s) — projected along F_hat
#         multiply by 1e4 in main loop to get cm²/V·s
def compute_mobility_tensor(kij, coords, box_lengths, h, h_inv, P, F_vec):
    F_mag_SI = np.linalg.norm(F_vec) * 1e2    # V/cm → V/m
    F_hat = F_vec / np.linalg.norm(F_vec)
    J = np.zeros(3)
    N = len(P)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            rij = coords[j] - coords[i]
            rij = minimum_image_triclinic(rij, h, h_inv)
            J += kij[i,j] * P[i] * (1 - P[j]) * (rij * angstrom_to_m)

    # PROJECT J onto field direction, then divide by scalar field magnitude
    # Previously returned the full vector J/|F|, causing sign/unit issues
    J_projected = np.dot(J, F_hat)      # scalar (m/s)
    mu_scalar = J_projected / F_mag_SI  # m²/V·s
    return mu_scalar

# MAIN LOOP
# write mobility file header once before the loop
# so every frame appends a row so  always have an up-to-date file
# even if the run is interrupted
mobility_outfile = "mobility_results.txt"
with open(mobility_outfile, "w") as f:
    f.write("# timestep   time_ns      mu_x(cm2/Vs)   mu_y(cm2/Vs)   "
            "mu_z(cm2/Vs)   mu_eff(cm2/Vs)\n")
print(f"Mobility results will be written to: {mobility_outfile}")

dump_file = "cg_beads.dump"
frames = parse_cg_dump(dump_file)
print("Total frames:", len(frames))

onsite_dict = load_onsite_energies_from_file(onsite_file, N)

print(f"Loaded onsite energies for {len(onsite_dict)} timesteps")

eff_mobility = []
time_frame = []

for timestep, box_lengths, h, h_inv, frame in frames:
    print(f"\n{'='*60}")
    print(f"Processing timestep: {timestep}")
    print(f"{'='*60}")

    # 2: Geometry
    frame_sorted = sorted(frame, key=lambda x: x[0])
    coords, normals = extract_coords_normals(frame_sorted)

    # 3: Intrachain transfer integrals
    dihedrals = compute_dihedrals_for_frame(frame_sorted)
    tphi = compute_tphi(dihedrals)
    tphi_chains = reshape_tphi_into_chains(tphi)

    # 4. Add onsite energy
    if timestep not in onsite_dict:
        raise ValueError(f"No onsite energies found for timestep {timestep}")

    onsite_energies = onsite_dict[timestep]

    # 5: Build Hamiltonian
    H = build_H_from_tphi(tphi_chains)
    H, pairs_added = add_through_space_to_H(H, coords, normals, box_lengths, h, h_inv)
    np.fill_diagonal(H, onsite_energies)     # onsite energies on diagonal

    # 8: Mobility tensor (average over 3 field directions)
    mu_cols = []
    for F_vec, label in zip(fields, field_labels):
        kij, eigvals, eigvecs, F_hat = build_kij(H, coords, box_lengths, h, h_inv, F_vec)
        P = solve_hole_populations(kij, Ndop=Ndop)
        mu_scalar = compute_mobility_tensor(kij, coords, box_lengths, h, h_inv, P, F_vec)
        mu_cols.append(mu_scalar)
        print(f"  μ_{label} = {mu_scalar*1e4:.6f} cm²/V·s")

    mu_x, mu_y, mu_z = [m * 1e4 for m in mu_cols]
    mu_eff = (mu_x + mu_y + mu_z) / 3
    time_ns_current = timestep * 0.1
    print(f"  μ_eff = {mu_eff:.6f} cm²/V·s")

    with open(mobility_outfile, "a") as f:
        f.write(f"{timestep:>10d}   {time_ns_current:>8.3f}   "
                f"{mu_x:>14.8f}   {mu_y:>14.8f}   "
                f"{mu_z:>14.8f}   {mu_eff:>14.8f}\n")

    eff_mobility.append(mu_eff)
    time_frame.append(timestep)

# organize and plot
eff_mobility = np.array(eff_mobility)
time_frame = np.array(time_frame)

#positive_mask = eff_mobility > 0
#eff_mobility = eff_mobility[positive_mask]
#time_frame = time_frame[positive_mask]

time_ns = time_frame * 0.1

mean_val = np.mean(eff_mobility)
std_val = np.std(eff_mobility)
sem = std_val / np.sqrt(len(eff_mobility))

plt.figure(figsize=(6, 4))
sc = plt.scatter(time_ns, eff_mobility, c=eff_mobility, s=50, alpha=0.7, cmap='viridis')
plt.errorbar(time_ns, eff_mobility, yerr=sem, fmt='none', capsize=3)
plt.axhline(0.0101, linestyle='--', color='r', label='Experiment = 0.0101')
plt.axhline(mean_val, linestyle='-', color='blue', label=f'Mean = {mean_val:.4f}')
plt.xlabel('Time (ns)')
plt.ylabel('Mobility (cm²/V·s)')
plt.legend()
plt.colorbar(sc, label='Mobility (cm²/V·s)')
plt.tight_layout()
plt.savefig("mobility_vs_time.png", dpi=300)
plt.close()

print(f"\nFinal: mean μ = {mean_val:.6f} ± {sem:.6f} cm²/V·s over {len(eff_mobility)} frames")
~                                                                                                                                                                                                                                                                                                                                                                  
~                                                                                                                                                                                  
