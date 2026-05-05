import os
import re
import sys
import shutil
import subprocess

#config
TEMPLATE_FILE     = "on_site_energy.input"
EQUIL_DATA_FILE   = "equilibrated.data"
FRAMES_DIR        = "frames"
RESULTS_DIR       = "results"
LAMMPS_EXE        = "/share/apps/images/run-lammps-20260104.bash"

MONOMER_TYPES     = [2, 3, 6]
ATOMS_PER_MONOMER = 5

os.makedirs(RESULTS_DIR, exist_ok=True)

# get frame index
if len(sys.argv) < 2:
    print("Usage: python on_site_energy.py <frame_index>")
    sys.exit(1)

frame_idx  = int(sys.argv[1])
frame_data = os.path.abspath(
    os.path.join(FRAMES_DIR, f"frame_{frame_idx:03d}.data"))
output_txt = os.path.join(
    RESULTS_DIR, f"onsite_energies_frame_{frame_idx:03d}.txt")

if not os.path.exists(frame_data):
    print(f"ERROR: {frame_data} not found. Run parse_frames.py first.")
    sys.exit(1)

print(f"[Frame {frame_idx:03d}] Using coordinates: {frame_data}")

# ── parse monomer atom IDs from eq3_last_equil.data ───────────────────────────
# Atom types 5/6/8 are the ring atoms; we group every 5 consecutive IDs
# into one monomer. This is stable across frames since topology never changes.

atom_ids_by_type = {t: [] for t in MONOMER_TYPES}

with open(EQUIL_DATA_FILE, "r") as f:
    lines = f.readlines()

in_atoms   = False
skip_blank = False

for line in lines:
    s = line.strip()
    if "Atoms" in line and "# full" in line:
        in_atoms   = True
        skip_blank = True
        continue
    if in_atoms:
        if skip_blank:
            if s == "": skip_blank = False
            continue
        if s == "":
            break
        parts = s.split()
        if len(parts) < 4:
            continue
        atom_id   = int(parts[0])
        atom_type = int(parts[2])
        if atom_type in MONOMER_TYPES:
            atom_ids_by_type[atom_type].append(atom_id)

atom_ids = sorted(aid for t in MONOMER_TYPES for aid in atom_ids_by_type[t])
monomer_atom_ids = [
    sorted(atom_ids[i:i + ATOMS_PER_MONOMER])
    for i in range(0, len(atom_ids), ATOMS_PER_MONOMER)
]
n_monomers = len(monomer_atom_ids)
print(f"[Frame {frame_idx:03d}] {n_monomers} monomers × "
      f"{ATOMS_PER_MONOMER} atoms each.")

# load template
with open(TEMPLATE_FILE, "r") as f:
    template_text = f.read()

# scratch dir for this frame's temp files
scratch = f"_scratch_frame_{frame_idx:03d}"
os.makedirs(scratch, exist_ok=True)

# loop over monomers
energies = []

for mono_idx, ids in enumerate(monomer_atom_ids, start=1):
    atom_id_str = " ".join(map(str, ids))

    input_text = (template_text
                  .replace("REPLACE_DATAFILE", frame_data)
                  .replace("REPLACE_IDS",      atom_id_str))

    tmp_input = os.path.join(scratch, f"mono_{mono_idx:04d}.in")
    tmp_log   = os.path.join(scratch, f"mono_{mono_idx:04d}.log")

    with open(tmp_input, "w") as f:
        f.write(input_text)

    #subprocess.run(
    #    [LAMMPS_EXE, "-in", tmp_input, "-log", tmp_log, "-screen", "none"],
    #    capture_output=True, text=True
    #)
    subprocess.run(
        [LAMMPS_EXE, "lmp", "-in", tmp_input, "-log", tmp_log],
        text=True
    )

    # parse E_MONOMER from log
    energy = None
    if os.path.exists(tmp_log):
        with open(tmp_log, "r") as lf:
            for line in lf:
                m = re.search(
                    r"E_MONOMER\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
                    line)
                if m:
                    energy = float(m.group(1))

    if energy is None:
        print(f"  [WARNING] frame {frame_idx}, monomer {mono_idx}: "
              f"parse failed — check {tmp_log}")
        energy = float("nan")

    energies.append(energy)

    if mono_idx % 25 == 0 or mono_idx == n_monomers:
        print(f"  [Frame {frame_idx:03d}] {mono_idx:>3}/{n_monomers}  "
              f"E = {energy:.6f} eV")

#write results
with open(output_txt, "w") as f:
    f.write("# frame_index   monomer_index   E_onsite_eV\n")
    for mono_idx, e in enumerate(energies, start=1):
        f.write(f"{frame_idx}\t{mono_idx}\t{e:.10f}\n")

print(f"[Frame {frame_idx:03d}] → {output_txt}")

# clean up scratch 
shutil.rmtree(scratch)
print(f"[Frame {frame_idx:03d}] Scratch cleaned.")
