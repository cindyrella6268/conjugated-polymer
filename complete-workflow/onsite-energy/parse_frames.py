import os
import numpy as np

EQUIL_DATA   = "equilibrated.data"
DUMP_FILE    = "equilibration.lammpstrj"
FRAMES_DIR   = "frames"

os.makedirs(FRAMES_DIR, exist_ok=True)

# parse static info from equilibrated data file

atom_info   = {}   
masses      = {}  
header_lines = [] 
masses_lines = []
in_atoms    = False
in_masses   = False
skip_blank  = False
found_masses = False

with open(EQUIL_DATA, "r") as f:
    lines = f.readlines()

# parse header (everything before Masses)
i = 0
while i < len(lines):
    line = lines[i]
    stripped = line.strip()

    if stripped.startswith("Masses"):
        in_masses  = True
        skip_blank = True
        i += 1
        continue

    if in_masses:
        if skip_blank:
            if stripped == "":
                skip_blank = False
            i += 1
            continue
        if stripped == "":
            in_masses = False
            i += 1
            continue
        parts = stripped.split()
        masses[int(parts[0])] = parts[1]   # type to mass
        i += 1
        continue

    if stripped.startswith("Atoms"):
        in_atoms   = True
        skip_blank = True
        i += 1
        continue

    if in_atoms:
        if skip_blank:
            if stripped == "":
                skip_blank = False
            i += 1
            continue
        if stripped == "":
            in_atoms = False
            i += 1
            continue
        parts = stripped.split()
        if len(parts) < 7:
            i += 1
            continue
        # full style: atom_id mol_id type charge x y z [ix iy iz]
        atom_id = int(parts[0])
        mol_id  = int(parts[1])
        atype   = int(parts[2])
        charge  = float(parts[3])
        atom_info[atom_id] = {"mol": mol_id, "type": atype, "charge": charge}
        i += 1
        continue

    i += 1

n_atoms      = len(atom_info)
n_atom_types = max(v["type"] for v in atom_info.values())
print(f"  Loaded {n_atoms} atoms, {n_atom_types} atom types, "
      f"{len(masses)} mass entries.")

# parse counts and box from header for writing
counts = {}          # {"atoms":3840, "bonds":4020, ...}
box    = {}          # xlo xhi ylo yhi zlo zhi xy xz yz
pair_coeffs_lines = []
in_pair = False
skip_pair_blank = False

for line in lines:
    s = line.strip()
    # counts
    for kw in ["atoms","bonds","angles","dihedrals","impropers",
               "atom types","bond types","angle types",
               "dihedral types","improper types"]:
        if s.endswith(kw):
            counts[kw] = int(s.split()[0])
    # box
    if "xlo xhi" in s and "xy" not in s:
        p = s.split(); box["xlo"]=p[0]; box["xhi"]=p[1]
    if "ylo yhi" in s:
        p = s.split(); box["ylo"]=p[0]; box["yhi"]=p[1]
    if "zlo zhi" in s:
        p = s.split(); box["zlo"]=p[0]; box["zhi"]=p[1]
    if "xy xz yz" in s:
        p = s.split(); box["xy"]=p[0]; box["xz"]=p[1]; box["yz"]=p[2]
    # pair coeffs
    if s.startswith("Pair Coeffs"):
        in_pair = True; skip_pair_blank = True
        pair_coeffs_lines.append(line)
        continue
    if in_pair:
        if skip_pair_blank:
            if s == "": skip_pair_blank = False
            pair_coeffs_lines.append(line)
            continue
        if s == "":
            in_pair = False
            pair_coeffs_lines.append(line)
            continue
        pair_coeffs_lines.append(line)

# parse dump frames
print(f"Reading trajectory from {DUMP_FILE} ...")

frames = []  
              
with open(DUMP_FILE, "r") as f:
    dlines = f.readlines()

j = 968
while j < len(dlines):
    s = dlines[j].strip()

    if s == "ITEM: TIMESTEP":
        timestep = int(dlines[j+1].strip())
        j += 2; continue

    if s == "ITEM: NUMBER OF ATOMS":
        n = int(dlines[j+1].strip())
        j += 2; continue

    if s.startswith("ITEM: BOX BOUNDS"):
        b = []
        for k in range(3):
            b.append(dlines[j+1+k].strip().split())
        box_dump = b
        j += 4; continue

    if s.startswith("ITEM: ATOMS"):
        col_names = s.replace("ITEM: ATOMS","").split()
        cidx = {name: k for k, name in enumerate(col_names)}
        coords = {}
        for k in range(n):
            parts = dlines[j+1+k].strip().split()
            aid = int(parts[cidx["id"]])
            xu  = float(parts[cidx["xu"]])
            yu  = float(parts[cidx["yu"]])
            zu  = float(parts[cidx["zu"]])
            coords[aid] = (xu, yu, zu)
        frames.append({"timestep": timestep,
                       "box_dump": box_dump,
                       "coords":   coords})
        j += 1 + n; continue

    j += 1

print(f"  Parsed {len(frames)} frames.")

# write one .data file per frame
print("Writing per-frame .data files ...")

for fi, frame in enumerate(frames):
    out_path = os.path.join(FRAMES_DIR, f"frame_{fi:03d}.data")

    # use box from dump (same triclinic system, but use dump's values)
    bd = frame["box_dump"]
    #xlo_b, xhi_b, xy_val = float(bd[0][0]), float(bd[0][1]), float(bd[0][2])
    #ylo_b, yhi_b, xz_val = float(bd[1][0]), float(bd[1][1]), float(bd[1][2])
    #zlo_b, zhi_b, yz_val = float(bd[2][0]), float(bd[2][1]), float(bd[2][2])

    xlo_b, xhi_b = float(bd[0][0]), float(bd[0][1])
    ylo_b, yhi_b = float(bd[1][0]), float(bd[1][1])
    zlo_b, zhi_b = float(bd[2][0]), float(bd[2][1])
    # convert to LAMMPS triclinic lo/hi
    xlo = xlo_b #- min(0.0)#, xy_val, xz_val, xy_val+xz_val)
    xhi = xhi_b #- max(0.0)#, xy_val, xz_val, xy_val+xz_val)
    ylo = ylo_b #- min(0.0)#, yz_val)
    yhi = yhi_b #- max(0.0)#, yz_val)
    zlo = zlo_b
    zhi = zhi_b

    coords = frame["coords"]

    with open(out_path, "w") as f:

        #header
        f.write(f"LAMMPS data file — frame {fi:03d} "
                f"(timestep {frame['timestep']})\n\n")
        #for kw in ["atoms","bonds","angles","dihedrals","impropers"]:
        #    if kw in counts:
        #        f.write(f"{counts[kw]} {kw}\n")
        #f.write("\n")
        #for kw in ["atom types","bond types","angle types",
        #           "dihedral types","improper types"]:
        #    if kw in counts:
        #        f.write(f"{counts[kw]} {kw}\n")
        #f.write("\n")
        f.write(f"{n_atoms} atoms\n\n")
        f.write(f"{n_atom_types} atom types\n\n")

        # box
        f.write(f"{xlo:.10f} {xhi:.10f} xlo xhi\n")
        f.write(f"{ylo:.10f} {yhi:.10f} ylo yhi\n")
        f.write(f"{zlo:.10f} {zhi:.10f} zlo zhi\n")
        #f.write(f"{xy_val:.10f} {xz_val:.10f} {yz_val:.10f} xy xz yz\n\n")

        # Masses
        f.write("Masses\n\n")
        for t in sorted(masses.keys()):
            f.write(f"{t} {masses[t]}\n")
        f.write("\n")

        # Pair Coeffs 
        for pl in pair_coeffs_lines:
            f.write(pl)

        # Atoms
        f.write("Atoms # full\n\n")
        for aid in sorted(atom_info.keys()):
            info = atom_info[aid]
            xu, yu, zu = coords[aid]
            f.write(f"{aid} {info['mol']} {info['type']} "
                    f"{info['charge']:.10f} "
                    f"{xu:.6f} {yu:.6f} {zu:.6f}\n")

    print(f"  Wrote {out_path}")

print("\nDone. Run on_site_energy.py next.")                                                                                                                             
