import numpy as np
import os

# constant parameters
kB = 8.617333262145e-5    # eV / K
hbar = 6.582119569e-16    # eV*s
angstrom_to_m = 1e-10     # m per Å

lambda1 = 0.45            # eV
T = 300.0                 # K
G = 0.005                 # dimensionless prefactor
LAM_MIN = 1e-8            # avoid division by 0

# box and mic
box = np.array([
    [-12.8061, 12.8061],
    [-20.0922, 20.0922],
    [-20.2903, 20.2903]
])
#box = np.array([
#    [-25.907350, 25.907350],
#    [-22.874629, 22.874629],
#    [-25.632754, 25.632754]
#])
box_lengths = box[:,1] - box[:,0]
use_min_image = True
def apply_minimum_image_A(vec, box_lengths_A):
    return vec - box_lengths_A * np.round(vec / box_lengths_A)

# input files
def load_inputs(H_file="H_onsite_w_sidechain.txt", coords_file="coords_wrapped.txt", p0_file="P0.txt"):
    H = np.loadtxt(H_file)
    coords = np.loadtxt(coords_file)
    P0 = np.loadtxt(p0_file) if os.path.exists(p0_file) else None
    return H, coords, P0

# kij
def build_kij_from_H(H, coords, F_vec):
    N = H.shape[0]
    eigvals, eigvecs = np.linalg.eigh(H)

    c2 = np.abs(eigvecs)**2
    #R_mo = compute_R_mo_with_MIC(c2, coords, box_lengths)
    R_mo = np.einsum("na,nk->ak", c2, coords)
    #if use_min_image:
        #R_mo = apply_minimum_image_A(R_mo, box_lengths)
    c4 = np.abs(eigvecs)**4
    sum_c4 = np.sum(c4, axis=0)

    lam_mat = lambda1 * (sum_c4.reshape((N,1)) + sum_c4.reshape((1,N)))
    np.fill_diagonal(lam_mat, 0.0)
    lam_safe = np.maximum(lam_mat, LAM_MIN)

    H_off = H.copy()
    np.fill_diagonal(H_off, 0.0)
    Vnm2 = H_off**2
    temp = np.tensordot(c2, Vnm2, axes=(0, 0))
    V2_mo = G**2 * (temp @ c2)
    V2_mo = 0.5 * (V2_mo + V2_mo.T)

    # Field setup
    F_mag = np.linalg.norm(F_vec)
    if F_mag == 0:
        raise ValueError("F_vec must be non-zero.")
    F_hat = F_vec / F_mag
    F_eV_per_A = F_mag * 1e-10

    prefactor = 2.0 * np.pi / hbar
    k_mat = np.zeros((N, N), dtype=float)

    for a in range(N):
        for b in range(N):
            if a == b:
                continue
            V2 = V2_mo[a, b]
            if V2 <= 0.0:
                continue
            lam_ab = lam_safe[a, b]
            E_a, E_b = eigvals[a], eigvals[b]

            rij = R_mo[b] - R_mo[a]
            #if use_min_image: 
                #rij = apply_minimum_image_A(rij, box_lengths)
            Rij_proj = np.dot(rij, F_hat)
            deltaG = (E_b - E_a) + (F_eV_per_A * Rij_proj)

            expo_arg = -((deltaG + lam_ab)**2) / (4.0 * lam_ab * kB * T)
            rate_pref = prefactor * V2 / np.sqrt(4.0 * np.pi * lam_ab * kB * T)
            k_mat[a, b] = rate_pref * np.exp(expo_arg)

    return k_mat, eigvals, eigvecs, R_mo, F_hat, F_mag

# Pi with gauss-seidel
def solve_hole_populations(kij, P0=None, Ndop=None, tol=1e-8, max_iter=100000, damping=0.5, verbose=True):
    kij = np.array(kij, dtype=float)
    np.fill_diagonal(kij, 0.0)
    N = kij.shape[0]

    if P0 is None:
        P = np.full(N, 1.0 / N, dtype=float)
    else:
        P = np.array(P0, dtype=float)

    if Ndop is None:
        Ndop = P.sum()
    sP = P.sum()
    if sP > 0:
        P *= (Ndop / sP)
    else:
        P[:] = Ndop / N

    KT = kij.T
    for it in range(1, max_iter + 1):
        P_old = P.copy()
        for i in range(N):
            R_i = kij[i].sum()
            if R_i <= 0:
                continue
            num_i = np.dot(KT[i], P)
            s_i = np.dot(kij[i] - KT[i], P)
            P_raw = (num_i / R_i) * (1.0 - s_i / R_i)
            if not np.isfinite(P_raw) or P_raw < 0:
                P_raw = 0.0
            P[i] = damping * P_raw + (1.0 - damping) * P[i]

        P *= Ndop / P.sum()
        residual = np.max(np.abs(P - P_old))
        if verbose and (it % 100 == 0 or residual < tol):
            print(f"Iter {it:6d}: residual = {residual:.3e}")
        if residual < tol:
            break
    return P

# mobility
def compute_mobility(kij, R_mo, P, F_hat, F_mag, Ndop, network_ID=None):
    R_mo_m = R_mo * angstrom_to_m
    box_lengths_m = box_lengths * angstrom_to_m

    J_vec = np.zeros(3, dtype=float)
    N = len(P)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            #skip hopping across networks
            if network_ID is not None and network_ID[i] != network_ID[j]:
                continue
            k_ij = kij[i, j]
            if k_ij <= 0.0:
                continue
            rij_m = R_mo_m[j] - R_mo_m[i]
            #if use_min_image: 
                #rij_m = apply_minimum_image_A(rij_m, box_lengths)
            J_vec += k_ij * P[i] * (1.0 - P[j]) * rij_m

    mu_along_field = np.dot(F_hat, J_vec) / (F_mag * Ndop)
    mu_tensor = np.outer(J_vec, F_hat) / (F_mag * Ndop)
    return mu_along_field, mu_tensor, J_vec

def run_single_field(H, coords, P0, network_ID, Ndop, F_vec, verbose=False):
    kij, eigvals, eigvecs, R_mo, F_hat, F_mag = build_kij_from_H(H, coords, F_vec)
    P = solve_hole_populations(kij, P0=P0, Ndop=Ndop, verbose=verbose)
    mu_along, mu_tensor, J_vec = compute_mobility(kij, R_mo, P, F_hat, F_mag, Ndop, network_ID=network_ID)
    return {
        "kij": kij,
        "P": P,
        "mu_tensor": mu_tensor,
        "mu_along": mu_along,
        "J_vec": J_vec,
        "F_hat": F_hat,
        "F_mag": F_mag,
        "R_mo": R_mo
    }

def compute_full_mobility_tensor(H, coords, P0=None, Ndop=1, network_ID=None, F_mag=20000.0, verbose=False):
    # three orthogonal fields (use same magnitude)
    fields = [
        np.array([F_mag, 0.0, 0.0]),
        np.array([0.0, F_mag, 0.0]),
        np.array([0.0, 0.0, F_mag])
    ]
    cols = []
    J_vectors = []
    results = []
    for F in fields:
        res = run_single_field(H, coords, P0, network_ID, Ndop, F, verbose=verbose)
        results.append(res)
        idx = int(np.argmax(np.abs(F)))  # 0,1,2
        col = res["mu_tensor"][:, idx]   # column vector
        cols.append(col)
        J_vectors.append(res["J_vec"])

    mu_full = np.column_stack(cols)  # assemble columns to get the full 3x3 mobility tensor
    return {
        "mu_full": mu_full,
        "results": results,
        "J_vectors": J_vectors
    }

# main
def main(verbose=False, F_mag=20000.0, Ndop=10):
    H, coords, P0 = load_inputs()
    network_ID = np.loadtxt("network_ID.txt", dtype=int)
    out = compute_full_mobility_tensor(H, coords, P0=P0, Ndop=Ndop, network_ID=network_ID, F_mag=F_mag, verbose=verbose)
    mu_full = out["mu_full"]
    print("\nfull mobility tensor [m^2 / V s]:\n", mu_full)

if __name__ == "__main__":
    main()
