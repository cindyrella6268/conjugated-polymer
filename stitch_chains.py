import numpy as np
import os

H_file = "H_onsite_11252025.txt" 
coords_file = "coords.txt"

box = np.array([
    [-12.80611686042665,  12.80611686042665],   
    [-20.09222334179953,  20.09222334179953], 
    [-20.290272100505682, 20.290272100505682] 
])

def wrap_into_box(coords, box):
    coords = np.asarray(coords)
    box_lo = box[:,0]
    box_hi = box[:,1]
    box_len = box_hi - box_lo
    wrapped = (coords - box_lo) % box_len + box_lo
    return wrapped

def center_coords_in_box(coords, box):
    box_center = np.mean(box, axis=1)
    return coords - box_center

def stitch_chain(coords, chain_indices, box):
    """
    say box is from -10 to 10 and i at 9, j at -9.
    distance would be 2, not 18
    """
    coords = coords.copy()
    box_len = box[:,1] - box[:,0]
    stitched = np.zeros((len(chain_indices), 3))
    stitched[0] = coords[chain_indices[0]]
    for i in range(1, len(chain_indices)):
        a = stitched[i-1]
        b = coords[chain_indices[i]]
        best, bestd = None, np.inf
        # search neighboring image offsets
        for kx in (-1,0,1):
            for ky in (-1,0,1):
                for kz in (-1,0,1):
                    b_image = b + np.array([kx,ky,kz]) * box_len
                    d = np.linalg.norm(b_image - a)
                    if d < bestd:
                        bestd = d
                        best = b_image
        stitched[i] = best
    out_coords = coords.copy()
    out_coords[chain_indices] = stitched
    return out_coords

def write_lammps_dump(filename, coords, eigvecs, box, atom_type=1, normalize_psi2=True):
    coords = np.asarray(coords)
    N = coords.shape[0]
    M = eigvecs.shape[1]
    coords_wrapped = wrap_into_box(coords, box)

    with open(filename, 'w') as f:
        for mo_idx in range(M):
            v = eigvecs[:, mo_idx]
            psi2 = np.abs(v)**2
            if normalize_psi2:
                s = psi2.sum()
                if s != 0:
                    psi2 = psi2 / s
            # write frame header
            f.write("ITEM: TIMESTEP\n")
            f.write(f"{mo_idx}\n")
            f.write("ITEM: NUMBER OF ATOMS\n")
            f.write(f"{N}\n")
            f.write("ITEM: BOX BOUNDS pp pp pp\n")
            f.write(f"{box[0,0]:.9f} {box[0,1]:.9f}\n")
            f.write(f"{box[1,0]:.9f} {box[1,1]:.9f}\n")
            f.write(f"{box[2,0]:.9f} {box[2,1]:.9f}\n")
            f.write("ITEM: ATOMS id type x y z psi2\n")
            for i in range(N):
                x,y,z = coords_wrapped[i]
                f.write(f"{i+1} {atom_type} {x:.6f} {y:.6f} {z:.6f} {psi2[i]:.6e}\n")

    print(f"Wrote {filename} with {M} frames and {N} atoms per frame.")

if __name__ == "__main__":
    H = np.loadtxt(H_file)
    coords = np.loadtxt(coords_file)
    if coords.ndim == 1:
        raise RuntimeError("coords.txt seems to have a single line — expected Nx3.")
    N = coords.shape[0]

    if H.shape[0] != H.shape[1] or H.shape[0] != N:
        print("Warning: H shape and coords length mismatch.")
        print(" H.shape:", H.shape, " coords N:", N)

    # diagonalize
    evals, evecs = np.linalg.eigh(H)   # columns are eigenvectors

    # reorder in ascending order
    order = np.argsort(evals)
    evecs = evecs[:, order]
    evals = evals[order]

    # make sure everything is inside the box
    coords_wrapped = wrap_into_box(coords, box)

    # write dump file
    outname = "mo_wrapped.dump"
    write_lammps_dump(outname, coords_wrapped, evecs, box)

    # centered version to look nicer
    coords_centered = center_coords_in_box(coords_wrapped, box)
    outname_centered = "mo_wrapped_centered.dump"
    box_centered = np.array([
        [box[0,0]-np.mean(box[0]), box[0,1]-np.mean(box[0])],
        [box[1,0]-np.mean(box[1]), box[1,1]-np.mean(box[1])],
        [box[2,0]-np.mean(box[2]), box[2,1]-np.mean(box[2])]
    ])
    write_lammps_dump(outname_centered, coords_centered, evecs, box_centered)

