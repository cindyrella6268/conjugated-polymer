import io
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from ase.geometry import find_mic
from ase.io.lammpsrun import read_lammps_dump_text

# physics parameters
J_inter_eV           = 0.1          # inter-chain coupling prefactor  (eV)
sigma                = 5.0          # length-scale parameter          (Å)
r0                   = 0.75 * sigma # reference distance              (Å)
alpha                = 1.0 / sigma  # exponential decay constant      (Å⁻¹)
through_space_cutoff = 15.0         # inter-chain pair cutoff         (Å)

kB    = 8.617333262145e-5   # Boltzmann constant       (eV K⁻¹)
hbar  = 6.582119569e-16     # reduced Planck constant  (eV·s)
ANG2M = 1e-10               # Ångström → metre

lambda1 = 0.45    # monomer reorganisation energy   (eV)
T       = 300.0   # temperature                     (K)
G       = 0.005   # dimensionless MO-coupling prefactor
LAM_MIN = 1e-8    # floor for reorganisation energy (eV)

Ndop  = 1         # number of holes
F_mag = 20000.0   # applied field magnitude         (V cm⁻¹)

# Polynomial coefficients for t(φ) = Σ_i C_i cos^i(φ)
C_poly = [-0.01, 1.275, 0.016, -0.870, -0.029, 0.540]

# Input files
ONSITE_FILE = "onsite_energies_all.txt"
DUMP_FILE   = "cg_beads.dump"

# Three orthogonal field directions for full tensor averaging
_FIELDS       = [np.array([F_mag, 0., 0.]),
                 np.array([0., F_mag, 0.]),
                 np.array([0., 0., F_mag])]
_FIELD_LABELS = ["x", "y", "z"]

# ASE-based LAMMPS dump reader
def _remap_dump_columns(raw_text: str) -> str:
    return re.sub(
        r'(?m)^(ITEM: ATOMS\s+id\s+)mol(\s+type\s+x\s+y\s+z\s+)nx(\s+)ny(\s+)nz',
        r'\1d_mol\2d_nx\3d_ny\4d_nz',
        raw_text
    )


def read_lammps_frames(dump_file: str) -> list:
    """
    Read every frame from a LAMMPS dump file via ASE.

    Each returned ASE Atoms object has:
      atoms.info['timestep']           LAMMPS timestep integer
      atoms.cell[:]                    3×3 cell matrix in Å (handles triclinic)
      atoms.pbc                        periodic boundary flags
      atoms.positions                  (N, 3) Cartesian coordinates in Å,
                                       sorted by atom-id (ASE default)
      atoms.arrays['type']             mol-id per atom (1-based int)
      atoms.arrays['d_nx/d_ny/d_nz']  normal-vector components
    """
    raw      = Path(dump_file).read_text()
    remapped = _remap_dump_columns(raw)

    # We only need geometry so 'C' is a harmless dummy element for all types.
    # Count distinct type values to build the required specorder list.
    first_block = remapped.split('ITEM: ATOMS', 1)[1] if 'ITEM: ATOMS' in remapped else ''
    type_values = re.findall(r'^\d+\s+(\d+)', first_block, re.MULTILINE)
    max_type    = max((int(v) for v in type_values), default=1)
    specorder   = ['C'] * max_type

    return list(
        read_lammps_dump_text(
            io.StringIO(remapped),
            index=':',
            specorder=specorder,
            order=True,     # sort atoms by id → consistent indexing
        )
    )


def load_onsite_energies(filename: str) -> dict:
    """
    Expected format (one entry per line):
        <timestep>  <monomer_id_1based>  <energy_eV>
    Returns
    dict  {timestep (int): {monomer_idx_0based (int): energy (float)}}
    """
    data: dict = {}
    with open(filename) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            ts, mon, energy = line.split()[:3]
            ts      = int(ts)
            mon_idx = int(mon) - 1          # convert to 0-based index
            energy  = float(energy)
            data.setdefault(ts, {})[mon_idx] = energy
    return data
  
# Chain topology  (generalised – no assumption about chain lengths)
def build_chain_topology(mol_ids: np.ndarray) -> dict:
    """
    Group atom indices by molecule id.

    Works for homopolymers, copolymers, or any mixture of chain lengths
    because it reads the mol-id directly from the dump instead of inferring
    it from a fixed beads-per-chain count.

    Parameters
    mol_ids : (N,) int array
        Per-atom molecule id (1-based) from atoms.arrays['type'].

    Returns
    dict  {mol_id (int): [atom_idx, ...]}
        Indices are in atom-id order (guaranteed by ASE's order=True).
    """
    chains: dict = defaultdict(list)
    for atom_idx, mol_id in enumerate(mol_ids):
        chains[int(mol_id)].append(atom_idx)
    return dict(sorted(chains.items()))

# Intrachain transfer integrals via backbone dihedral angles
def _align_normals(normals: list) -> list:
    aligned = [np.array(normals[0])]
    for n in normals[1:]:
        n = np.array(n)
        aligned.append(n if np.dot(aligned[-1], n) >= 0.0 else -n)
    return aligned


def _t_phi(angle_deg: float, C: list) -> float:
    cos_phi = np.cos(np.radians(angle_deg))
    return sum(c * cos_phi**i for i, c in enumerate(C))


def compute_intrachain_couplings(chain_topology: dict,
                                 normals: np.ndarray,
                                 C: list = C_poly) -> dict:
    result = {}
    for mol_id, indices in chain_topology.items():
        aligned = _align_normals([normals[i] for i in indices])
        couplings = []
        for k in range(len(aligned) - 1):
            cos_theta = float(np.clip(np.dot(aligned[k], aligned[k + 1]), -1.0, 1.0))
            couplings.append(_t_phi(np.degrees(np.arccos(cos_theta)), C))
        result[mol_id] = couplings
    return result

# Hamiltonian construction
def _normalise_rows(v: np.ndarray) -> np.ndarray:
    """Normalise each row of an (N, 3) array; zero rows are left as zero."""
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return v / norms

def build_hamiltonian(chain_topology: dict,
                      intrachain_couplings: dict,
                      coords: np.ndarray,
                      normals: np.ndarray,
                      cell: np.ndarray,
                      pbc: np.ndarray,
                      onsite_energies: np.ndarray):
    N = len(coords)
    H = np.zeros((N, N))

    # --- intrachain off-diagonal elements ---
    for mol_id, indices in chain_topology.items():
        for k, t in enumerate(intrachain_couplings[mol_id]):
            i, j = indices[k], indices[k + 1]
            H[i, j] += t
            H[j, i] += t

    # --- interchain through-space elements ---
    normals_n = _normalise_rows(normals.copy())

    # atom → chain map for O(1) intrachain filtering
    atom_chain = np.empty(N, dtype=int)
    for mol_id, indices in chain_topology.items():
        for idx in indices:
            atom_chain[idx] = mol_id

    # Compute ALL pairwise displacements with a single find_mic call.
    # all_disp[i, j] = MIC-corrected vector from site i to site j.
    raw_disp         = coords[np.newaxis, :, :] - coords[:, np.newaxis, :]  # (N, N, 3)
    mic_disp, mic_d  = find_mic(raw_disp.reshape(-1, 3), cell, pbc=pbc)
    mic_disp         = mic_disp.reshape(N, N, 3)   # (N, N, 3)
    mic_d            = mic_d.reshape(N, N)          # (N, N) distances

    pairs_added = 0
    for i in range(N):
        for j in range(i + 1, N):
            if atom_chain[i] == atom_chain[j]:
                continue
            r = mic_d[i, j]
            if r > through_space_cutoff:
                continue
            r_ij  = mic_disp[i, j]
            r_hat = r_ij / r
            fn, fm = normals_n[i], normals_n[j]
            orient = (np.dot(fn, r_hat)**2
                      * np.dot(fm, r_hat)**2
                      * np.dot(fn, fm)**2)
            w = J_inter_eV * orient * np.exp(-alpha * (r - r0))
            if abs(w) > 1e-12:
                H[i, j] += w
                H[j, i] += w
                pairs_added += 1

    # onsite energies
    np.fill_diagonal(H, onsite_energies)
    return H, pairs_added

# Marcus-theory rate matrix
def _mo_centers_mic(c2: np.ndarray,
                    coords: np.ndarray,
                    cell: np.ndarray,
                    pbc: np.ndarray) -> np.ndarray:
    
    N_sites, N_MOs = c2.shape
    anchors        = np.argmax(c2, axis=0)              # (N_MOs,)  index of max site

    # For every (MO, site) pair compute the displacement from the anchor site.
    # disp[a, n] = r_n − r_{anchor_a}  before MIC,  shape (N_MOs, N_sites, 3)
    disp_raw = coords[np.newaxis, :, :] - coords[anchors, np.newaxis, :]

    # apply find_mic in one vectorised call
    mic_disp, _ = find_mic(disp_raw.reshape(-1, 3), cell, pbc=pbc)
    mic_disp    = mic_disp.reshape(N_MOs, N_sites, 3)  # (N_MOs, N_sites, 3)

    # weighted average displacement from each anchor, then shift back
    # c2.T has shape (N_MOs, N_sites); einsum sums over sites
    R_rel = np.einsum('an,ans->as', c2.T, mic_disp)    # (N_MOs, 3)
    R_mo  = R_rel + coords[anchors]                     # (N_MOs, 3)
    return R_mo


def build_rate_matrix(H: np.ndarray,
                      coords: np.ndarray,
                      cell: np.ndarray,
                      pbc: np.ndarray,
                      F_vec: np.ndarray):
    N = H.shape[0]
    eigvals, eigvecs = np.linalg.eigh(H)        # eigh exploits symmetry of H

    c2 = eigvecs ** 2                            # |c_{na}|²  (N_sites, N_MOs)
    c4 = eigvecs ** 4                            # |c_{na}|⁴

    # --- MO centres (vectorised MIC) ---
    R_mo = _mo_centers_mic(c2, coords, cell, pbc)   # (N_MOs, 3)

    # --- reorganisation energy matrix ---
    ipr = c4.sum(axis=0)                         # IPR_a = Σ_n |c_{na}|⁴  (N_MOs,)
    lam = lambda1 * (ipr[:, None] + ipr[None, :])
    np.fill_diagonal(lam, 0.0)
    lam = np.maximum(lam, LAM_MIN)

    # --- effective coupling squared in MO basis ---
    H_off  = H - np.diag(np.diag(H))
    V2_mo  = G**2 * (c2.T @ (H_off**2) @ c2)    # (N_MOs, N_MOs)
    V2_mo  = 0.5 * (V2_mo + V2_mo.T)             # symmetrise numerical noise

    # --- field ---
    F_mag_loc = float(np.linalg.norm(F_vec))
    F_hat     = F_vec / F_mag_loc
    F_eV_A    = F_mag_loc * 1e-8                 # V cm⁻¹ → eV Å⁻¹

    # --- MO-centre displacements with vectorised MIC ---
    # r_ab_raw[a, b] = R_mo[b] - R_mo[a]  before MIC
    r_ab_raw        = R_mo[np.newaxis, :, :] - R_mo[:, np.newaxis, :]  # (N, N, 3)
    r_ab_mic, _     = find_mic(r_ab_raw.reshape(-1, 3), cell, pbc=pbc)
    r_ab_mic        = r_ab_mic.reshape(N, N, 3)  # (N, N, 3)

    # field projection for all pairs: shape (N, N)
    field_proj = np.einsum('ijk,k->ij', r_ab_mic, F_hat) * F_eV_A

    # --- assemble rate matrix ---
    # ΔG[a,b] = ε_b − ε_a + F·r_ab
    delta_G  = (eigvals[np.newaxis, :] - eigvals[:, np.newaxis]) + field_proj

    # Marcus exponent: −(ΔG + λ)² / (4 λ k_B T)
    kBT4     = 4.0 * kB * T
    exponent = -((delta_G + lam) ** 2) / (kBT4 * lam)

    # prefactor: (2π/ℏ) · V² / √(4π λ k_B T)
    prefac   = (2.0 * np.pi / hbar) * V2_mo / np.sqrt(np.pi * kBT4 * lam)

    kij      = prefac * np.exp(exponent)
    np.fill_diagonal(kij, 0.0)
    kij[V2_mo <= 0.0] = 0.0                      # zero coupling → zero rate

    return kij, eigvals, R_mo, F_hat, F_mag_loc

# Master-equation steady-state hole populations
def solve_hole_populations(kij: np.ndarray,
                            Ndop: int = Ndop,
                            tol: float = 1e-8,
                            max_iter: int = 100_000,
                            damping: float = 0.5) -> np.ndarray:
   
    kij = np.array(kij, dtype=float)
    np.fill_diagonal(kij, 0.0)
    N  = kij.shape[0]
    P  = np.full(N, Ndop / N, dtype=float)
    KT = kij.T

    for _ in range(max_iter):
        P_old = P.copy()
        for i in range(N):
            R_i = kij[i].sum()
            if R_i <= 0.0:
                continue
            num_i = np.dot(KT[i], P)
            s_i   = np.dot(kij[i] - KT[i], P)
            P_raw = (num_i / R_i) * (1.0 - s_i / R_i)
            if not np.isfinite(P_raw) or P_raw < 0.0:
                P_raw = 0.0
            P[i]  = damping * P_raw + (1.0 - damping) * P[i]
        P *= Ndop / P.sum()
        if np.max(np.abs(P - P_old)) < tol:
            break

    return P

# Mobility tensor
def compute_mobility_column(kij: np.ndarray,
                             R_mo: np.ndarray,
                             P: np.ndarray,
                             F_hat: np.ndarray,
                             F_mag_loc: float,
                             cell: np.ndarray,
                             pbc: np.ndarray,
                             Ndop: int = Ndop) -> np.ndarray:

    N = len(P)

    # MIC displacements for all MO pairs in one call
    r_raw       = R_mo[np.newaxis, :, :] - R_mo[:, np.newaxis, :]   # (N, N, 3)
    r_mic, _    = find_mic(r_raw.reshape(-1, 3), cell, pbc=pbc)
    r_mic       = r_mic.reshape(N, N, 3) * ANG2M                     # Å → m

    # weight matrix: k_{ab} P_a (1−P_b)
    W = kij * P[:, None] * (1.0 - P[None, :])       # (N, N)
    np.fill_diagonal(W, 0.0)

    # J_α = Σ_{ab} W_{ab} Δr_{ab,α}   →  einsum over (N, N) pairs
    J_vec = np.einsum('ab,abc->c', W, r_mic)         # (3,)

    F_SI  = F_mag_loc * 1e2                          # V cm⁻¹ → V m⁻¹
    return J_vec / (F_SI * Ndop)                     # m² V⁻¹ s⁻¹


def compute_full_mobility_tensor(H: np.ndarray,
                                  coords: np.ndarray,
                                  cell: np.ndarray,
                                  pbc: np.ndarray,
                                  Ndop: int = Ndop) -> tuple:
    cols = []
    for F_vec in _FIELDS:
        kij, _, R_mo, F_hat, F_mag_loc = build_rate_matrix(
            H, coords, cell, pbc, F_vec)
        P      = solve_hole_populations(kij, Ndop=Ndop)
        mu_col = compute_mobility_column(
            kij, R_mo, P, F_hat, F_mag_loc, cell, pbc, Ndop=Ndop)
        cols.append(mu_col)

    mu_cm  = np.column_stack(cols) * 1e4    # m² V⁻¹ s⁻¹ → cm² V⁻¹ s⁻¹
    mu_eff = np.trace(mu_cm) / 3.0
    return mu_cm, mu_eff, mu_cm[0, 0], mu_cm[1, 1], mu_cm[2, 2]

# Main
def main():
    out_file = "mobility_results.txt"
    with open(out_file, "w") as fh:
        fh.write("# timestep   time_ns      mu_x(cm2/Vs)   mu_y(cm2/Vs)   "
                 "mu_z(cm2/Vs)   mu_eff(cm2/Vs)\n")
    print(f"Results → {out_file}")

    # --- read all frames with ASE ---
    print(f"Reading {DUMP_FILE} ...")
    frames = read_lammps_frames(DUMP_FILE)
    print(f"  {len(frames)} frames loaded")

    # --- onsite energies ---
    onsite_all = load_onsite_energies(ONSITE_FILE)
    print(f"  Onsite energies for {len(onsite_all)} timesteps")

    eff_mobilities: list = []
    timesteps:      list = []

    for atoms in frames:
        ts = atoms.info['timestep']
        print(f"\n{'='*60}\nTimestep {ts}\n{'='*60}")

        # Geometry (ASE has already sorted by atom-id)
        coords  = atoms.positions                        # (N, 3) Å
        cell    = np.array(atoms.cell)                   # (3, 3) Å
        pbc     = atoms.pbc                              # (3,) bool
        normals = np.column_stack([atoms.arrays['d_nx'],
                                   atoms.arrays['d_ny'],
                                   atoms.arrays['d_nz']])  # (N, 3)
        mol_ids = atoms.arrays['type']                   # (N,) 1-based mol-id
        N       = len(coords)

        # Onsite energies (0-based index; missing → 0 eV)
        if ts not in onsite_all:
            raise ValueError(f"No onsite energies for timestep {ts}")
        onsite_map   = onsite_all[ts]
        onsite_array = np.array([onsite_map.get(i, 0.0) for i in range(N)])

        # Chain topology – works for any chain-length distribution
        chain_topo = build_chain_topology(mol_ids)
        chain_lens = [len(v) for v in chain_topo.values()]
        print(f"  N={N}, chains={len(chain_topo)}, "
              f"lengths: {min(chain_lens)}–{max(chain_lens)}")

        # Intrachain transfer integrals from dihedral angles
        intra = compute_intrachain_couplings(chain_topo, normals)

        # Build site Hamiltonian
        H, n_pairs = build_hamiltonian(
            chain_topo, intra, coords, normals, cell, pbc, onsite_array)
        print(f"  Through-space pairs: {n_pairs}")

        # Full 3×3 mobility tensor
        mu_cm, mu_eff, mu_x, mu_y, mu_z = compute_full_mobility_tensor(
            H, coords, cell, pbc, Ndop=Ndop)

        t_ns = ts * 0.1
        print(f"  μ_x   = {mu_x:.6f} cm²/V·s")
        print(f"  μ_y   = {mu_y:.6f} cm²/V·s")
        print(f"  μ_z   = {mu_z:.6f} cm²/V·s")
        print(f"  μ_eff = {mu_eff:.6f} cm²/V·s")
        evals_mu = np.sort(np.linalg.eigvalsh(0.5 * (mu_cm + mu_cm.T)))[::-1]
        print(f"  Principal mobilities: {evals_mu}")

        with open(out_file, "a") as fh:
            fh.write(f"{ts:>10d}   {t_ns:>8.3f}   "
                     f"{mu_x:>14.8f}   {mu_y:>14.8f}   "
                     f"{mu_z:>14.8f}   {mu_eff:>14.8f}\n")

        eff_mobilities.append(mu_eff)
        timesteps.append(ts)

    # --- summary & plot ---
    mu_arr = np.array(eff_mobilities)
    t_ns   = np.array(timesteps) * 0.1
    mean   = mu_arr.mean()
    sem    = mu_arr.std() / np.sqrt(len(mu_arr))
    print(f"\nFinal: μ = {mean:.6f} ± {sem:.6f} cm²/V·s  ({len(mu_arr)} frames)")

    plt.figure(figsize=(6, 4))
    sc = plt.scatter(t_ns, mu_arr, c=mu_arr, s=50, alpha=0.7, cmap='viridis')
    plt.errorbar(t_ns, mu_arr, yerr=sem, fmt='none', capsize=3)
    plt.axhline(0.0101, ls='--', color='r', label='Experiment = 0.0101')
    plt.axhline(mean,   ls='-',  color='b', label=f'Mean = {mean:.4f}')
    plt.xlabel('Time (ns)')
    plt.ylabel('Mobility (cm²/V·s)')
    plt.legend()
    plt.colorbar(sc, label='Mobility (cm²/V·s)')
    plt.tight_layout()
    plt.savefig("mobility_vs_time.png", dpi=300)
    plt.close()
    print("Plot → mobility_vs_time.png")


if __name__ == "__main__":
    main()
