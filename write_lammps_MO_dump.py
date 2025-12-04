import numpy as np

coords = np.loadtxt("coords.txt")
H = np.loadtxt("H_onsite_11252025.txt")

box = np.array([
    [-12.80611686042665,  12.80611686042665],
    [-20.09222334179953,  20.09222334179953],
    [-20.290272100505682, 20.290272100505682]
])

box_lo = box[:,0]
box_hi = box[:,1]
box_lengths = box_hi - box_lo

# get MO center
def get_MO_center_MIC(c2_a, coords, box_lengths):
    i_ref = np.argmax(c2_a)     # reference center:index of monomer with highest psi2 weight
    r_shift = coords - coords[i_ref]     # shift coordinates such that reference monomer is at 000
    r_shift -= box_lengths * np.round(r_shift / box_lengths)     # mic in each axis: how many box lengths each displacement spans
    R_rel = np.sum(c2_a[:, None] * r_shift, axis=0)    # where MO is relative to i_ref
    R_abs = R_rel + coords[i_ref]       # convert relative COM to absolute COM inside the simulation coords
    R_abs = (R_abs - (-box_lengths/2)) % box_lengths + (-box_lengths/2) #wrap back into box
    return R_abs

def write_MO_dump(H, coords, box, box_lengths, outfile):
    N = H.shape[0]

    # diagnolize
    eigvals, eigvecs = np.linalg.eigh(H)

    c2 = np.abs(eigvecs)**2

    with open(outfile, "w") as f:

        for a in range(N):   # MO index
            psi2 = c2[:, a]

            R_mo = get_MO_center_MIC(psi2, coords, box_lengths)

            f.write("ITEM: TIMESTEP\n")
            f.write(f"{a}\n")

            f.write("ITEM: NUMBER OF ATOMS\n")
            f.write(f"{N+1}\n")   # +1 bead for COM

            f.write("ITEM: BOX BOUNDS pp pp pp\n")
            f.write(f"{box[0][0]} {box[0][1]}\n")
            f.write(f"{box[1][0]} {box[1][1]}\n")
            f.write(f"{box[2][0]} {box[2][1]}\n")

            f.write("ITEM: ATOMS id type x y z psi2\n")

            # Original monomers
            for i in range(N):
                x, y, z = coords[i]
                f.write(f"{i+1} 1 {x:.8f} {y:.8f} {z:.8f} {psi2[i]:.8e}\n")

            # COM bead
            f.write(f"{N+1} 99 {R_mo[0]:.8f} {R_mo[1]:.8f} {R_mo[2]:.8f} -1\n")


    print("✓ MO dump file written to:", outfile)

write_MO_dump(H, coords, box, box_lengths, "mo_wrapped_w_COM2.dump")
