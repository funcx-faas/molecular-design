"""Functions used to perform chemistry tasks """
import os
from typing import List
from io import StringIO

import numpy as np
import pandas as pd
from ase.io import read
from ase.optimize import LBFGSLineSearch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from xtb.ase.calculator import XTB

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline

"""SIMULATION FUNCTIONS: Quantum chemistry parts of the workflow"""
def generate_initial_xyz(mol_string: str) -> str:
    """Generate the XYZ coordinates for a molecule.
    
    Args:
        mol_string: SMILES string

    Returns:
        - InChI string for the molecule
        - XYZ coordinates for the molecule
    """

    # Generate 3D coordinates for the molecule
    mol = Chem.MolFromSmiles(mol_string)
    if mol is None:
        raise ValueError(f'Parse failure for {mol_string}')
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=1)
    AllChem.MMFFOptimizeMolecule(mol)

    # Save geometry as 3D coordinates
    xyz = f"{mol.GetNumAtoms()}\n"
    xyz += mol_string + "\n"
    conf = mol.GetConformer()
    for i, a in enumerate(mol.GetAtoms()):
        s = a.GetSymbol()
        c = conf.GetAtomPosition(i)
        xyz += f"{s} {c[0]} {c[1]} {c[2]}\n"

    return xyz


def compute_vertical(smiles: str) -> float:
    """Run the ionization potential computation

    Args:
        smiles: SMILES string to evaluate
    Returns:
        Ionization energy in Ha
    """

    # Make the initial geometry
    xyz = generate_initial_xyz(smiles)
    
    # Make the XTB calculator
    calc = XTB(accuracy=0.05)
    
    # Parse the molecule
    atoms = read(StringIO(xyz), format='xyz')

    # Compute the neutral geometry
    #  Uses QCEngine (https://github.com/MolSSI/QCEngine) to handle interfaces to XTB
    atoms.calc = calc
    dyn = LBFGSLineSearch(atoms, logfile=None)
    dyn.run(fmax=0.02, steps=250)
    
    neutral_energy = atoms.get_potential_energy()

    # Compute the energy of the relaxed geometry in charged form
    charges = np.ones((len(atoms),)) * (1 / len(atoms))
    atoms.set_initial_charges(charges)
    charged_energy = atoms.get_potential_energy()
    
    return charged_energy - neutral_energy



"""MACHINE LEARNING FUNCTIONS: Predicting the output of quantum chemistry"""

def compute_morgan_fingerprints(smiles: str, fingerprint_length: int, fingerprint_radius: int):
    """Get Morgan Fingerprint of a specific SMILES string.
    Adapted from: <https://github.com/google-research/google-research/blob/
    dfac4178ccf521e8d6eae45f7b0a33a6a5b691ee/mol_dqn/chemgraph/dqn/deep_q_networks.py#L750>
    Args:
      graph (str): The molecule as a SMILES string
      fingerprint_length (int): Bit-length of fingerprint
      fingerprint_radius (int): Radius used to compute fingerprint
    Returns:
      np.array. shape = [hparams, fingerprint_length]. The Morgan fingerprint.
    """
    # Parse the molecule
    molecule = Chem.MolFromSmiles(smiles)

    # Compute the fingerprint
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(
        molecule, fingerprint_radius, fingerprint_length)
    arr = np.zeros((1,), dtype=bool)

    # ConvertToNumpyArray takes ~ 0.19 ms, while
    # np.asarray takes ~ 4.69 ms
    DataStructs.ConvertToNumpyArray(fingerprint, arr)
    return arr


class MorganFingerprintTransformer(BaseEstimator, TransformerMixin):
    """Class that converts SMILES strings to fingerprint vectors"""

    def __init__(self, length: int = 256, radius: int = 4):
        self.length = length
        self.radius = radius

    def fit(self, X, y=None):
        return self  # Do need to do anything

    def transform(self, X, y=None):
        """Compute the fingerprints
        
        Args:
            X: List of SMILES strings
        Returns:
            Array of fingerprints
        """
        fps = []
        for x in X: 
            fps.append(compute_morgan_fingerprints(x, self.length, self.radius))
            
        return fps     
        

if __name__ == "__main__":
    energy = compute_vertical('OC')
    print(energy)
