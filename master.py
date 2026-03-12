# master code to compute mobility for many time steps with coarse-grained dump file

import numpy as np
import csv
from typing import List, Tuple
import os
import matplotlib.pyplot as plt
import time

n_chains = 20
n_monomers_per_chain = 10
N = n_chains * n_monomers_per_chain

epsilon_default = 0.0

J_inter_eV = 0.1
sigma = 5.0
r0 = 0.75 * sigma
alpha = 1.0 / sigma
through_space_cutoff = 15.0

kB = 8.617333262145e-5    # eV / K
hbar = 6.582119569e-16    # eV*s
angstrom_to_m = 1e-10     # m per Å

lambda1 = 0.45            # eV
T = 300.0                 # K
G = 0.005                 # dimensionless prefactor
LAM_MIN = 1e-8            # avoid division by 0

Ndop = 1
F_mag = 20000             #V/cm
    
# parse dump file
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

            box_lengths = np.array([
                xhi - xlo,
                yhi - ylo,
                zhi - zlo
            ])

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
        frames.append((current_timestep, box_lengths, current_frame))

    return frames

# write a data file for each frame
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

# get coords and normals
def extract_coords_normals(frame):
    coords = []
    normals = []

    for (_, _, coord, normal, _, _, _) in frame:
        coords.append(coord)
        normals.append(normal)

    return np.array(coords), np.array(normals)

# align normals
def align_normals_consistently(normals):
    aligned = [normals[0]]
    for i in range(1, len(normals)):
        if np.dot(aligned[-1], normals[i]) < 0:
            aligned.append(-normals[i])
        else:
            aligned.append(normals[i])
    return aligned

# compute dihedrals
def compute_dihedrals_for_frame(
    frame,
    n_chains=20,
    beads_per_chain=10
):
    chains = [[] for _ in range(n_chains)]

    for idx, (_, _, _, normal, _, _, _) in enumerate(frame):
        chain_id = idx // beads_per_chain
        chains[chain_id].append(normal)

    dihedrals = []

    for chain_id, normals in enumerate(chains):
        normals = align_normals_consistently(normals)

        for i in range(len(normals) - 1):
            cos_theta = np.clip(
                np.dot(normals[i], normals[i + 1]),
                -1.0, 1.0
            )
            angle_deg = np.degrees(np.arccos(cos_theta))
            dihedrals.append((chain_id, i, angle_deg))

    return dihedrals

# tphi calculation
C = [-0.01, 1.275, 0.016, -0.870, -0.029, 0.540]

def t_phi_from_angle(angle_deg, C):
    phi_rad = np.radians(angle_deg)
    cos_phi = np.cos(phi_rad)
    return sum(c * (cos_phi**i) for i, c in enumerate(C))

def compute_tphi(dihedrals, C=C):
    angles = [angle for (_, _, angle) in dihedrals]
    tphi_values = [t_phi_from_angle(a, C) for a in angles]
    return np.array(tphi_values)

def reshape_tphi_into_chains(tphi,
                              n_chains=20,
                              beads_per_chain=10):

    dihed_per_chain = beads_per_chain - 1

    chains = []
    for c in range(n_chains):
        start = c * dihed_per_chain
        end = start + dihed_per_chain
        chains.append(list(tphi[start:end]))

    return chains

# onsite energy
def load_onsite_energies(path):
    energies = np.zeros(N, dtype=float)
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split() 
            mon_id = int(parts[0])
            e_val = float(parts[1]) 
            energies[mon_id - 1] = e_val
    return energies
    
def combine_onsite_results(n_monomers):

    energies = np.zeros(n_monomers)

    for i in range(1, n_monomers + 1):

        fname = f"result_{i}.txt"

        with open(fname) as f:
            parts = f.read().split()
            energies[i-1] = float(parts[1])

    return energies
    
# build hamiltonian
def build_H_from_tphi(tphi_chains, epsilon=epsilon_default):

    n_dihed = len(tphi_chains[0])
    n_per_chain = n_dihed + 1
    N_local = len(tphi_chains) * n_per_chain

    H = np.zeros((N_local, N_local))
    np.fill_diagonal(H, epsilon)

    for c, chain_vals in enumerate(tphi_chains):
        for m, t in enumerate(chain_vals):
            i = c * n_per_chain + m
            j = c * n_per_chain + m + 1
            H[i, j] = t
            H[j, i] = t

    return H

def normalize_vectors(v):
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return v / norms


def compute_w_nm(fn, fm, rvec):
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

def minimum_image(rij_vec, box_lengths):
    return rij_vec - np.rint(rij_vec / box_lengths) * box_lengths

def add_through_space_to_H(H, coords, normals, box_lengths):

    normals = normalize_vectors(normals)
    N_local = coords.shape[0]

    pairs_added = 0

    for i in range(N_local):
        for j in range(i+1, N_local):

            rij_vec = coords[j] - coords[i]
            rij_vec = minimum_image(rij_vec, box_lengths)

            r = np.linalg.norm(rij_vec)
            if r > through_space_cutoff:
                continue

            # skip intrachain
            if i // n_monomers_per_chain == j // n_monomers_per_chain:
                continue

            w = compute_w_nm(normals[i], normals[j], rij_vec)

            if abs(w) > 1e-12:
                H[i, j] += w
                H[j, i] += w
                pairs_added += 1

    return H, pairs_added

def build_kij(H, coords, box_lengths, F_vec):
    N = H.shape[0]
    eigvals, eigvecs = np.linalg.eigh(H)
    c2 = np.abs(eigvecs)**2
    c4 = np.abs(eigvecs)**4
    sum_c4 = np.sum(c4, axis=0)
    lam = lambda1*(sum_c4.reshape((N,1))+sum_c4.reshape((1,N)))
    np.fill_diagonal(lam,0)
    lam = np.maximum(lam,LAM_MIN)
    H_off = H.copy()
    np.fill_diagonal(H_off,0)
    V2 = H_off**2
    temp = np.tensordot(c2, V2, axes=(0,0))
    V2_mo = G**2 * (temp @ c2)
    V2_mo = 0.5*(V2_mo+V2_mo.T)
    F_hat = F_vec/np.linalg.norm(F_vec)
    F_eV_A = np.linalg.norm(F_vec)*1e-10
    pref = 2*np.pi/hbar
    kij = np.zeros((N,N))

    for a in range(N):
        for b in range(N):
            if a==b: continue
            if V2_mo[a,b]<=0: continue

            rij = minimum_image(coords[b]-coords[a], box_lengths)
            deltaG = (eigvals[b]-eigvals[a]) + F_eV_A*np.dot(rij,F_hat)

            expo = -((deltaG+lam[a,b])**2)/(4*lam[a,b]*kB*T)
            rate_pref = pref*V2_mo[a,b]/np.sqrt(4*np.pi*lam[a,b]*kB*T)
            kij[a,b] = rate_pref*np.exp(expo)

    return kij, eigvals, eigvecs, F_hat
    
def solve_hole_populations(kij, Ndop=Ndop, tol=1e-8, max_iter=100000):
    kij = np.array(kij)
    np.fill_diagonal(kij,0)
    N = kij.shape[0]
    P = np.full(N, Ndop/N)
    KT = kij.T

    for _ in range(max_iter):
        P_old = P.copy()
        for i in range(N):
            R_i = kij[i].sum()
            if R_i<=0: continue
            num_i = np.dot(KT[i],P)
            P[i] = num_i/R_i
        P *= Ndop/P.sum()
        if np.max(np.abs(P-P_old))<tol:
            break

    return P

def compute_mobility_tensor(kij, coords, box_lengths, P, F_vec):
    F_hat = F_vec/np.linalg.norm(F_vec)
    J = np.zeros(3)

    N=len(P)

    for i in range(N):
        for j in range(N):
            if i==j: continue
            rij = coords[j]-coords[i]
            rij = minimum_image(rij, box_lengths)
            J += kij[i,j]*P[i]*(1-P[j])*(rij*angstrom_to_m)

    mu_col = J/np.linalg.norm(F_vec)
    return mu_col

# main
dump_file = "cg_beads.dump"
frames = parse_cg_dump(dump_file)
results = []
print("Total frames:", len(frames))

fields = [np.array([F_mag,0,0]), np.array([0,F_mag,0]), np.array([0,0,F_mag])]

eff_mobility = []
time_frame = []

for timestep, box_lengths, frame in frames:
    mu_cols = []
    print("\nProcessing timestep:", timestep)
    frame_sorted = sorted(frame,key=lambda x:x[0])
    coords, normals = extract_coords_normals(frame_sorted)
    dihedrals = compute_dihedrals_for_frame(frame_sorted)
    tphi = compute_tphi(dihedrals)
    tphi_chains = reshape_tphi_into_chains(tphi)

    data_file = f"frame_{timestep}.data"
    write_frame_as_data(frame_sorted, "eq3_last_equil.data", data_file)
    os.system(f"python on_site_energy.py {data_file}")
    os.system("sbatch submit_run_all.sh")
    while True:
        files = [f for f in os.listdir() if f.startswith("result_")]
        if len(files) == N:
            break
        time.sleep(5)
    onsite_energies = combine_onsite_results(N)
    H = build_H_from_tphi(tphi_chains)
    H, pairs_added = add_through_space_to_H(H, coords, normals, box_lengths)
    np.fill_diagonal(H, onsite_energies)
    os.system("rm result_*.txt")

    # mobility
    for F_vec in fields:
        kij, eigvals, eigvecs, F_hat = build_kij(H,coords,box_lengths,F_vec)
        P = solve_hole_populations(kij,Ndop=Ndop)
        mu_col = compute_mobility_tensor(kij,coords,box_lengths,P,F_vec)
        mu_cols.append(mu_col)

    mu_full = np.column_stack(mu_cols)*1e4
    mu_eff = np.trace(mu_full)/3

    print("Timestep:",timestep)
    print("μ_eff:",mu_eff)

    eff_mobility.append(mu_eff)
    time_frame.append(timestep)

eff_mobility = np.array(eff_mobility)
time_frame = np.array(time_frame)
# keep only positive mobilities for plotting
positive_mask = eff_mobility > 0
eff_mobility = eff_mobility[positive_mask]
time_frame = time_frame[positive_mask]
# convert frames to ns
time_ns = time_frame * 0.1
colors = eff_mobility
size = 50
plt.figure(figsize=(6,4))
plt.scatter(time_ns, eff_mobility, c=colors, s=size, alpha=0.7, cmap='viridis')
# statistics
mean_val = np.mean(eff_mobility)
std_val = np.std(eff_mobility)
sem = std_val / np.sqrt(len(eff_mobility))
plt.axhline(0.0101, linestyle='--', color='r', label='experiment = 0.0101')
plt.axhline(mean_val, linestyle='-', color='blue', label='Average')
plt.errorbar(time_ns, eff_mobility, yerr=sem, fmt='none', capsize=3)
plt.xlabel('Time (ns)')
plt.ylabel('Mobility (cm$^2$/V·s)')
plt.legend()
plt.colorbar(label='Mobility value')
plt.tight_layout()
plt.savefig("mobility_vs_time.png", dpi=300)
plt.close()

