from torch.utils.data import Dataset
import numpy as np
import torch
import random
import os
from tqdm import tqdm

class Dockground_Dataset(Dataset):

    def __init__(self,data_path, use_list):
        listfiles=os.listdir(data_path)
        file_list=[]            
        count_check=0
        for item in tqdm(listfiles):
            if item[:-4] not in use_list:
                continue
            count_check+=1
            tmp_path=os.path.join(data_path,item)
            file_list.append(tmp_path)
            
        self.listfiles=file_list


    def __getitem__(self, idx):
        file_path=self.listfiles[idx]
        data=np.load(file_path)
        return data


    def __len__(self):
        return len(self.listfiles)