lmp=/scratch/projects/depablolab/cindy/applications/build/lmp
output_file=onsite_backbone_eV.txt

echo "# monomer_id   E_elec_eV" > $output_file

for f in onsite_mono_*.in; do
    echo "Running $f ..."

    # Run LAMMPS and capture output
    log=$(mktemp)
    $lmp -in "$f" > "$log"

    # extract value printed like:  E_MONOMER = -0.12345 eV
    energy=$(grep "E_MONOMER" "$log" | awk '{print $3}')

    # extract monomer index from filename
    id=$(echo "$f" | sed 's/onsite_mono_\([0-9]*\).in/\1/')

    echo "$id  $energy" >> $output_file
    rm "$log"
done
