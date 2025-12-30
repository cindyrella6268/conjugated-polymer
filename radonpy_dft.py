import sys
from radonpy.core import utils, poly
from radonpy.ff.gaff2_mod import GAFF2_mod
from radonpy.sim import qm
from radonpy.sim.preset import eq, tc
import os
import pickle
smiles1= "C(F)(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F"
smiles2= "CCn1cc[n+](C)c1"
temp = 300
press = 1.0
omp_psi4 = 48
mpi = 48
omp = 1
gpu = 0
mem = 64000
work_dir = '.'
ff = GAFF2_mod()
def rescale_charges(mol, scale_factor=0.8, charge_props=("RESP", "ESP", "AtomicCharge")):
    net_charges_before = {prop: sum(float(atom.GetProp(prop)) for atom in mol.GetAtoms() if atom.HasProp(prop)) for prop in charge_props}
    for atom in mol.GetAtoms():
        for prop in charge_props:
            if atom.HasProp(prop):
                charge = float(atom.GetProp(prop))
                atom.SetProp(prop, str(charge * scale_factor))
    net_charges_after = {prop: sum(float(atom.GetProp(prop)) for atom in mol.GetAtoms() if atom.HasProp(prop)) for prop in charge_props}
    for prop in charge_props:
        print(f"{prop} net charge before scaling: {net_charges_before[prop]:.6f}")
        print(f"{prop} net charge after scaling: {net_charges_after[prop]:.6f}")
if __name__ == '__main__':
    if os.path.exists('mol1.pkl'):
        with open('mol1.pkl', 'rb') as f:
            mol1 = pickle.load(f)
        with open('mol2.pkl', 'rb') as f:
            mol2 = pickle.load(f)
        rescale_charges(mol1)
        rescale_charges(mol2)
    else:
        mol1 = utils.mol_from_smiles(smiles1)
        mol2 = utils.mol_from_smiles(smiles2)
        mol1, energy = qm.conformation_search(mol1, ff=ff , work_dir=work_dir, psi4_omp=omp_psi4, mpi=mpi, omp=omp, memory=mem, log_name='monomer1')
        mol2, energy = qm.conformation_search(mol2, ff=ff , work_dir=work_dir, psi4_omp=omp_psi4, mpi=mpi, omp=omp, memory=mem, log_name='monomer2')
        qm.assign_charges(mol1, charge='RESP', charge_basis='6-31+G(d,p)', opt=True, work_dir=work_dir, omp=omp_psi4, memory=mem, log_name='monomer1')
        with open('mol1.pkl', 'wb') as f:
            pickle.dump(mol1, f)
        qm.assign_charges(mol2, charge='RESP', charge_basis='6-31+G(d,p)', opt=True, work_dir=work_dir, omp=omp_psi4, memory=mem, log_name='monomer2')
        with open('mol2.pkl', 'wb') as f:
            pickle.dump(mol2, f)
        rescale_charges(mol1)
        rescale_charges(mol2)
    # Force field assignment
    result = ff.ff_assign(mol1)
    result = ff.ff_assign(mol2)
    if not result:
        print('[ERROR: Can not assign force field parameters.]')
    mols = [mol1, mol2]
    n = [500,500]
    # Generate mixture cell
    ac = poly.amorphous_mixture_cell(mols, n, density=0.005)
    # Equilibrate using radonpy
    eqmd = eq.EQ21step(ac, work_dir=work_dir)
    ac = eqmd.exec(temp=temp, press=1.0, mpi=mpi, omp=omp, gpu=gpu)
