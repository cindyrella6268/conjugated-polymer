# get dihedral angles in .txt file
import numpy as np

def parse_cg_dump(dump_file):
    frames = []
    current_frame = []
    reading_atoms = False

    with open(dump_file, 'r') as f:
        for line in f:
            if line.startswith("ITEM: TIMESTEP"):
                if current_frame:
                    frames.append(current_frame)
                    current_frame = []
            elif line.startswith("ITEM: ATOMS"):
                reading_atoms = True
            elif line.startswith("ITEM:"):
                reading_atoms = False
            elif reading_atoms:
                parts = line.strip().split()
                if len(parts) < 11:
                    print(f"Warning: skipped malformed line: {line}")
                    continue
                atom_id = int(parts[0])
                mol_type = int(parts[1])
                x, y, z = map(float, parts[2:5])
                nx, ny, nz = map(float, parts[5:8])
                ix, iy, iz = map(int, parts[8:11])
                current_frame.append((atom_id, mol_type, np.array([x, y, z]), np.array([nx, ny, nz]), ix, iy, iz))

    if current_frame:
        frames.append(current_frame)

    return frames

def align_normals_consistently(normals):
    aligned = [normals[0]]
    for i in range(1, len(normals)):
        prev = aligned[-1]
        curr = normals[i]
        if np.dot(curr, prev) < 0:
            curr = -curr
        aligned.append(curr)
    return aligned

def compute_dihedrals_and_save(dump_file, output_txt="dihedrals_531.txt"):
    frames = parse_cg_dump(dump_file)
    frame = frames[0]  # take first snapshot

    # Determine number of chains and monomers automatically
    num_monomers = len(frame)
    # assume all chains have same length as first chain
    chain_length_guess = 10  # replace with actual if known, otherwise auto-detect
    num_chains = num_monomers // chain_length_guess

    chains = [[] for _ in range(num_chains)]

    # collect normals for each chain
    for idx, (_, _, _, normal, _, _, _) in enumerate(frame):
        chain_id = idx // chain_length_guess
        chains[chain_id].append(normal)

    # align normals along each chain
    flipped_normals = [align_normals_consistently(normals) for normals in chains]

    # compute dihedral angles (angle between consecutive normals)
    all_angles = []
    for chain_normals in flipped_normals:
        for i in range(len(chain_normals) - 1):
            n1 = chain_normals[i]
            n2 = chain_normals[i + 1]
            cos_angle = np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # avoid numerical errors
            angle_deg = np.degrees(np.arccos(cos_angle))
            all_angles.append(angle_deg)

    # save to TXT
    with open(output_txt, 'w') as f_txt:
        for chain_id, chain_normals in enumerate(flipped_normals):
            f_txt.write(f"Chain {chain_id}:\n")
            for i in range(len(chain_normals) - 1):
                angle = all_angles[chain_id * (len(chain_normals) - 1) + i]
                f_txt.write(f"  Dihedral {i}: {angle:.2f} degrees\n")
            f_txt.write("\n")

    print(f"Saved dihedral angles to '{output_txt}'")
    return all_angles

if __name__ == "__main__":
    compute_dihedrals_and_save("cg_beads531.dump")
