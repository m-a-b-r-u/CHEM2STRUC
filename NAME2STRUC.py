import tkinter as tk
from tkinter import simpledialog
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import requests
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

atom_data = {
    1: {'color': 'white', 'radius': 0.32},   # Hydrogen
    2: {'color': 'lightcyan', 'radius': 0.46},  # Helium
    3: {'color': 'violet', 'radius': 1.33},  # Lithium
    4: {'color': 'palegreen', 'radius': 1.02},  # Beryllium
    5: {'color': 'peachpuff', 'radius': 0.82},  # Boron
    6: {'color': 'black', 'radius': 0.75},  # Carbon
    7: {'color': 'blue', 'radius': 0.71},  # Nitrogen
    8: {'color': 'red', 'radius': 0.63},  # Oxygen
    9: {'color': 'limegreen', 'radius': 0.64},  # Fluorine
    10: {'color': 'lightskyblue', 'radius': 0.67},  # Neon
    11: {'color': 'darkviolet', 'radius': 1.55},  # Sodium
    12: {'color': 'lime', 'radius': 1.39},  # Magnesium
    13: {'color': 'lightgrey', 'radius': 1.26},  # Aluminum
    14: {'color': 'tan', 'radius': 1.16},  # Silicon
    15: {'color': 'orange', 'radius': 1.11},  # Phosphorus
    16: {'color': 'yellow', 'radius': 1.03},  # Sulfur
    17: {'color': 'green', 'radius': 0.99},  # Chlorine
    18: {'color': 'lightcyan', 'radius': 0.96},  # Argon
}

def draw_sphere(ax, center, radius, color, res=100):
    u = np.linspace(0, 2 * np.pi, res)
    v = np.linspace(0, np.pi, res)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=0.75, rstride=2, cstride=2, edgecolor='none')

def get_smiles(name):
    urls = [
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/CanonicalSMILES/TXT",
        f"https://cactus.nci.nih.gov/chemical/structure/{name}/smiles",
        f"https://www.chemspider.com/smiles/{name}.txt",
        f"https://www.ebi.ac.uk/chebi/ws/rest/compound/search?searchCategory=CHEMICAL_NAME&search={name}"
    ]
    for url in urls:
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.text.strip()
        except requests.RequestException:
            continue
    return None

def plot_3d(mol, ax):
    conf = mol.GetConformer()
    positions = [conf.GetAtomPosition(atom.GetIdx()) for atom in mol.GetAtoms()]
    numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]

    for pos, num in zip(positions, numbers):
        atom_info = atom_data.get(num, {'color': 'grey', 'radius': 1.20})
        draw_sphere(ax, (pos.x, pos.y, pos.z), atom_info['radius'], atom_info['color'])

    for bond in mol.GetBonds():
        start_pos = conf.GetAtomPosition(bond.GetBeginAtomIdx())
        end_pos = conf.GetAtomPosition(bond.GetEndAtomIdx())
        ax.plot([start_pos.x, end_pos.x], [start_pos.y, end_pos.y], [start_pos.z, end_pos.z], color='grey', lw=2.0)

    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')

    all_positions = np.array(positions)
    x_min, x_max = all_positions[:, 0].min(), all_positions[:, 0].max()
    y_min, y_max = all_positions[:, 1].min(), all_positions[:, 1].max()
    z_min, z_max = all_positions[:, 2].min(), all_positions[:, 2].max()

    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    margin = max_range * 0.05
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_zlim(z_min - margin, z_max + margin)

    ax.set_box_aspect([1, 1, 1])
    ticks = np.linspace(min(x_min, y_min, z_min), max(x_max, y_max, z_max), num=5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)

    ax.grid(False)
    ax.set_facecolor('white')
    ax.figure.patch.set_facecolor('white')

def show_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print("Invalid SMILES string.")
        return
    mol_with_hs = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol_with_hs, randomSeed=1)
    AllChem.MMFFOptimizeMolecule(mol_with_hs)

    fig = plt.figure(figsize=(12, 6))

    ax2d = fig.add_subplot(121)
    img = Draw.MolToImage(mol, size=(1000, 1000))
    ax2d.imshow(img)
    ax2d.axis('off')

    ax3d = fig.add_subplot(122, projection='3d')
    plot_3d(mol_with_hs, ax3d)

    plt.show()

def prompt_name():
    root = tk.Tk()
    root.withdraw()
    return simpledialog.askstring("Input", "Enter chemical name:")

def main():
    name = prompt_name()
    if name:
        smiles = get_smiles(name)
        if smiles:
            print(f"SMILES for {name}: {smiles}")
            show_molecule(smiles)
        else:
            print(f"Failed to retrieve SMILES for {name}")
    else:
        print("No chemical name provided.")

if __name__ == "__main__":
    main()
