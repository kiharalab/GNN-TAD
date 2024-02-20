import os
import sys
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm

from sklearn.preprocessing import add_dummy_feature
# from data_processing.Extract_Interface import Extract_Interface
from rdkit.Chem.rdmolfiles import MolFromPDBFile
import numpy as np
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from Preprocessing.feature_processing_atom import get_atom_feature
from scipy.spatial import distance_matrix
from Bio.PDB.PDBParser import PDBParser

Residue_id = {
    "GLY": 1,
    "ALA": 2,
    "VAL": 3,
    "LEU": 4,
    "ILE": 5,
    "PRO": 6,
    "PHE": 7,
    "TYR": 8,
    "TRP": 9,
    "SER": 10,
    "THR": 11,
    "CYS": 12,
    "MET": 13,
    "ASN": 14,
    "GLN": 15,
    "ASP": 16,
    "GLU": 17,
    "LYS": 18,
    "ARG": 19,
    "HIS": 20,
}


def Prepare_Input(onehot = 0, charge = 0, structure_path="", output_dir=""):
    id = os.path.split(structure_path)[1].split(".")[0]
    pdb_path = structure_path
    
    structure_mol = MolFromPDBFile(os.path.abspath(pdb_path), sanitize=False)
    # ligand_mol = MolFromPDBFile(ligand_path, sanitize=False)
    atom_count = structure_mol.GetNumAtoms()
    # ligand_count = ligand_mol.GetNumAtoms()
    structure_feature = get_atom_feature(structure_mol)
    
    
    c1 = structure_mol.GetConformers()[0]
    d1 = np.array(c1.GetPositions())
    adj1 = GetAdjacencyMatrix(structure_mol) + np.eye(atom_count)


    #extract position and residue type
    charge = []
    add_feature = []
    if charge:
        with open(structure_path,'r') as file:
            lines = file.readlines()
        for line in lines:
            items = [i for i in line.split(" ") if len(i)>0]
            if len(items) > 1:
                if line.startswith("ATOM") and not items[2].startswith("H") and "OXT" not in items[2]:
                    charge.append(np.array([float(line.split(" ")[-2])]))
    
    p = PDBParser()
    structure = p.get_structure("s",pdb_path)
    
    res_num = 0
    for model in structure:
        for chain in model:
            for res in chain:
                res_id = Residue_id[res.get_resname()]
                for atom in res:
                    res_num = res.get_id()[1]
                    if onehot:
                        add_feature.append(np.append(np.array([int(i == res_num) for i in range(30)]),np.array([int(i == res_id - 1) for i in range(20)])))
                    else:
                        add_feature.append(np.array([res_num,res_id/20]))
    if not onehot:
        nor_pos = np.array(add_feature)[:,0]/res_num
    
        if charge:
            add_feature = np.concatenate((np.expand_dims(nor_pos,1), np.expand_dims(np.array(add_feature)[:,1],1),charge),1)
        else:
            add_feature = np.concatenate((np.expand_dims(nor_pos,1), np.expand_dims(np.array(add_feature)[:,1],1),charge),1)

    else:
        if charge:
            add_feature = np.concatenate((np.array(add_feature),charge),1)

    

    
    H = np.concatenate([structure_feature, np.array(add_feature)], 1) # 1-28: atom features, 29-58: position, 59 to 78: residue 
    adj2 = distance_matrix(d1, d1)
    
    try:
        os.mkdir(output_dir)
    except:
        pass
    input_file=os.path.join(output_dir,id+".npz")
    
    np.savez(input_file,  H=H, A1=adj1, A2=adj2)
    
    return input_file
    


def call_prepare(input_dir, output_dir):
    paths = os.listdir(input_dir)
    for path in tqdm(paths):
        path = os.path.join(input_dir,path)
        #print(path)
        #print(output_dir)
        #print(label_path)
        Prepare_Input(1,0,path,output_dir)
#call_prepare(p1,p2,p3)