import numpy as np
import os

def extract_backbone_atoms_all_frames(filepath, selected_types={2, 3, 6}):
    frames = []
    box_bounds = []
    headers = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        if "ITEM: TIMESTEP" in lines[i]:
            timestep = int(lines[i + 1].strip())
            i += 2
        elif "ITEM: NUMBER OF ATOMS" in lines[i]:
            num_atoms = int(lines[i + 1].strip())
            i += 2
        elif "ITEM: BOX BOUNDS" in lines[i]:
            box_bounds = []
            for j in range(3):
                parts = lines[i + 1 + j].split()
                box_bounds.append((float(parts[0]), float(parts[1])))
            i += 4
        elif "ITEM: ATOMS" in lines[i]:
            headers = lines[i].strip().split()[2:]
            frame_atoms = []
            i += 1
            while i < len(lines) and not lines[i].startswith("ITEM:"):
                parts = lines[i].strip().split()
                atom_id = int(parts[0])
                atom_type = int(parts[1])
                x, y, z = map(float, parts[2:5])
                if atom_type in selected_types:
                    frame_atoms.append([atom_id, atom_type, x, y, z])
                i += 1
            frames.append((timestep, frame_atoms))
        else:
            i += 1

    return frames, box_bounds, headers

def write_backbone_dump_all_frames(output_path, frames, box_bounds, headers, wrap=True):
    with open(output_path, 'w') as f:
        for timestep, atoms in frames:
           # atoms = wrap_coordinates(atoms, box_bounds)
            f.write(f"ITEM: TIMESTEP\n{timestep}\n")
            f.write(f"ITEM: NUMBER OF ATOMS\n{len(atoms)}\n")
            f.write("ITEM: BOX BOUNDS pp pp pp\n")
            for bound in box_bounds:
                f.write(f"{bound[0]} {bound[1]}\n")
            f.write(f"ITEM: ATOMS id type x y z\n")
            for atom in atoms:
                f.write(" ".join(map(str, atom)) + "\n")

def main():
    input_file = 'equilibration.lammpstrj'
    output_file = 'backbone_only.dump'

    frames, box_bounds, headers = extract_backbone_atoms_all_frames(input_file, selected_types={2, 3, 6})
    write_backbone_dump_all_frames(output_file, frames, box_bounds, headers, wrap = False)

if __name__ == '__main__':
    main()
