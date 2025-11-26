import subprocess
import os
import numpy as np

lmp_path = "/scratch/projects/depablolab/cindy/applications/build/lmp" #

template_file = "on_site_energy.input"

data_file = "eq3_last_equil.data"

monomer_types = [5, 6, 8]
atoms_per_monomer = 5

atom_ids_by_type = {t: [] for t in monomer_types}

# parse data file
with open(data_file, "r") as f:
    lines = f.readlines()

read_atoms = False
for line in lines:
    if "Atoms" in line:
        read_atoms = True
        skip_blank = True
        continue
    if read_atoms:
        if skip_blank:
            if line.strip() == "":
                skip_blank = False
                continue
            else:
                continue
        if line.strip() == "":
            break
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        atom_id = int(parts[0])
        atom_type = int(parts[2])
        if atom_type in monomer_types:
            atom_ids_by_type[atom_type].append(atom_id)

atom_ids = sorted([aid for t in monomer_types for aid in atom_ids_by_type[t]])
print(f"Found {len(atom_ids)} ring atoms total (types {monomer_types})")

monomer_atom_ids = []
for i in range(0, len(atom_ids), atoms_per_monomer):
    monomer_atom_ids.append(sorted(atom_ids[i:i + atoms_per_monomer]))

print(f"Detected {len(monomer_atom_ids)} monomers based on ring atoms.")
#print(f"Detected {n_monomers} monomers (based on types 5,6,8)")

#input file for lammps
energies = []

for i, atom_ids in enumerate(monomer_atom_ids, start=1):
    with open(template_file, "r") as f:
        input_text = f.read()

    atom_id_str = " ".join(map(str, atom_ids))
    input_text = input_text.replace("REPLACE_IDS", atom_id_str)

    temp_input = f"onsite_mono_{i}.in"
    with open(temp_input, "w") as f:
        f.write(input_text)

    # diagnostics
    print(f">>> Running monomer {i}, atom IDs: {atom_ids}")

    # singularity to use lammps
    sing_cmd = (
    "singularity exec --overlay "
    "/scratch/yl10749/my_env/overlay-15GB-500K.ext3:ro "
    "/scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif "
    f"/bin/bash -c 'source /ext3/env.sh; /scratch/projects/depablolab/cindy/applications/build/lmp -in {temp_input}'"
    )

    #run lammps
    result = subprocess.run(sing_cmd, shell=True, capture_output=True, text=True)
    #diagnostics
    print(result.stdout)
    print(result.stderr)

    E = None
    for line in result.stdout.split("\n"):
        if "E_MONOMER" in line:
            try:
                E = float(line.split('=')[1].split()[0])
                energies.append(E)
            except:
                pass
            break

energies = np.array(energies)
np.savetxt("onsite_energies_eV.txt", energies, fmt="%.6f")
