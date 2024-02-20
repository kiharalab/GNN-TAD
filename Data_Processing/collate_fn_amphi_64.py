import numpy as np
import torch

def collate_fn_amphi_64(batch):
    max_natoms = max([len(item['H']) for item in batch if item is not None])
    H = np.zeros((len(batch), max_natoms, 84))
    A1 = np.zeros((len(batch), max_natoms, max_natoms))
    A2 = np.zeros((len(batch), max_natoms, max_natoms))


    Atoms_Number=[]
    for i in range(len(batch)):
        natom = len(batch[i]['H'])
        H[i, :natom] = batch[i]['H']
        A1[i, :natom, :natom] = batch[i]['A1']
        A2[i, :natom, :natom] = batch[i]['A2']

        Atoms_Number.append(natom)
    H = torch.from_numpy(H).float()
    A1 = torch.from_numpy(A1).float()
    A2 = torch.from_numpy(A2).float()

    Atoms_Number=torch.Tensor(Atoms_Number)

    return H, A1, A2, Atoms_Number #, keys