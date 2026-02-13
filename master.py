# master code to compute mobility for many time steps with coarse-grained dump file

import numpy as np
import csv
from typing import List, Tuple

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

# onsite energy


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


# main
dump_file = "cg_beads.dump"

frames = parse_cg_dump(dump_file)

print("Total frames:", len(frames))

for timestep, box_lengths, frame in frames:

    print("\nProcessing timestep:", timestep)

    coords, normals = extract_coords_normals(frame)

    dihedrals = compute_dihedrals_for_frame(frame)

    tphi = compute_tphi(dihedrals)

    tphi_chains = reshape_tphi_into_chains(tphi)

    H = build_H_from_tphi(tphi_chains)

    H, pairs_added = add_through_space_to_H(H, coords, normals, box_lengths)

    print("Hamiltonian size:", H.shape)
    print("Through-space pairs added:", pairs_added)
