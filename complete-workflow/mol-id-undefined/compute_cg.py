import numpy as np
from collections import defaultdict

mass_dict = {
    1: 1.007825,   # O
    2: 12.011,   # C
    3: 12.011,   # O
    4: 15.999,   # C
    5: 12.011,   # C
    6: 32.067,   # C
    7: 1.008,  # H
    8: 1.008,   # S
    9: 12.011,    # H
    10: 15.999,    # H
    11: 1.008
}

def parse_dump(filepath, selected_types={2, 3, 6}):
    frames = []
    box_bounds = None
    with open(filepath, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        if "ITEM: TIMESTEP" in lines[i]:
            timestep = int(lines[i + 1])
            i += 2
        elif "ITEM: NUMBER OF ATOMS" in lines[i]:
            i += 2
        elif "ITEM: BOX BOUNDS" in lines[i]:
            box_bounds = []
            for j in range(3):
                parts = lines[i + 1 + j].split()
                box_bounds.append((float(parts[0]), float(parts[1])))
            i += 4
        elif "ITEM: ATOMS" in lines[i]:
            i += 1
            atoms = []
            while i < len(lines) and not lines[i].startswith("ITEM:"):
                parts = lines[i].strip().split()
                atom_id = int(parts[0])
                atom_type = int(parts[1])
                x, y, z = map(float, parts[2:5])
                if atom_type in selected_types:
                    atoms.append((atom_id, atom_type, np.array([x, y, z])))
                i += 1
            frames.append(atoms)
        else:
            i += 1
    return frames, box_bounds

def compute_image_flags(com, box_bounds):
    ix, iy, iz = 0, 0, 0
    for i, (coord, (lo, hi)) in enumerate(zip(com, box_bounds)):
        L = hi - lo
        delta = coord - lo
        if i == 0:
            ix = int(np.floor(delta / L))
        elif i == 1:
            iy = int(np.floor(delta / L))
        else:
            iz = int(np.floor(delta / L))
    return ix, iy, iz

def compute_COM_and_orientation(monomer_atoms):
    # monomer_atoms: list of (type, position)
    total_mass = 0.0
    com = np.zeros(3)
    masses = []

    for atom_type, pos in monomer_atoms:
        m = mass_dict[atom_type]
        total_mass += m
        com += m * pos
        masses.append(m)

    com /= total_mass

    # Shift to COM frame
    shifted_positions = [pos - com for _, pos in monomer_atoms]

    # Compute inertia tensor
    I = np.zeros((3, 3))
    #for mass, (_, pos) in zip(masses, monomer_atoms):
    for mass, (atom_type, pos) in zip(masses, monomer_atoms):
        r = pos - com
        I += mass * (np.dot(r, r) * np.identity(3) - np.outer(r, r))

    # Diagonalize inertia tensor
    eigvals, eigvecs = np.linalg.eigh(I)

    # Get the principal axis with smallest inertia (normal to ring)
    normal_vector = eigvecs[:, np.argmax(eigvals)]
    return com, normal_vector

def process_frames(frames, box_bounds, atoms_per_monomer=5):
    cg_beads_all_frames = []

    for frame in frames:
        cg_beads = []

        # Sort frame atoms by atom ID to preserve order
        frame_sorted = sorted(frame, key=lambda x: x[0])  # x[0] is atom_id

        for i in range(0, len(frame_sorted), atoms_per_monomer):
            group = frame_sorted[i:i+atoms_per_monomer]
            if len(group) == atoms_per_monomer:
                monomer_atoms = [(atom_type, pos) for _, atom_type, pos in group]
                com, normal = compute_COM_and_orientation(monomer_atoms)
                ix, iy, iz = compute_image_flags(com, box_bounds)
                cg_beads.append((i // atoms_per_monomer, com, normal, ix, iy, iz))

        cg_beads_all_frames.append(cg_beads)

    return cg_beads_all_frames

#def save_output(cg_beads_all_frames, output_file='cg_beads_min.xyz'):
#    with open(output_file, 'w') as f:
#        for frame in cg_beads_all_frames:
#            f.write(f"{len(frame)}\n")
#           f.write("Atoms with COM and orientation (ring normal)\n")
#            for mol_id, com, normal in frame:
#                x, y, z = com
#                nx, ny, nz = normal
#                f.write(f"CGB {x:.4f} {y:.4f} {z:.4f} {nx:.4f} {ny:.4f} {nz:.4f}\n")

def save_output(cg_beads_all_frames, box_bounds, output_file='cg_beads.dump'):
    with open(output_file, 'w') as f:
        for timestep, frame in enumerate(cg_beads_all_frames):
            f.write("ITEM: TIMESTEP\n")
            f.write(f"{timestep}\n")
            f.write("ITEM: NUMBER OF ATOMS\n")
            f.write(f"{len(frame)}\n")
            f.write("ITEM: BOX BOUNDS pp pp pp\n")
            for bound in box_bounds:
                f.write(f"{bound[0]} {bound[1]}\n")
            f.write("ITEM: ATOMS id type x y z nx ny nz ix iy iz\n")
            for i, (mol_id, com, normal, ix, iy, iz) in enumerate(frame, start=1):
                x, y, z = com
                nx, ny, nz = normal
                f.write(f"{i} 1 {x:.4f} {y:.4f} {z:.4f} {nx:.4f} {ny:.4f} {nz:.4f} {ix} {iy} {iz}\n")

def main():
    frames, box_bounds = parse_dump("backbone_only.dump")
    # check if total atoms per frame is multiple of group size
    if any(len(f) % 5 != 0 for f in frames):
        print("Warning: some frames have leftover atoms not grouped in 5s.")
    cg_beads_all_frames = process_frames(frames, box_bounds)
    save_output(cg_beads_all_frames, box_bounds)

if __name__ == '__main__':
    main()
