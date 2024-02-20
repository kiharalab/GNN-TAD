from cmath import inf
import os
from GNN.GNN_Model_atom import GNN_Model_atom
from GNN.GNN_Model_res import GNN_Model_res
from ops.train_utils import count_parameters,initialize_model
import torch
from Data_Processing.Dockground_Dataset import Dockground_Dataset
from torch.utils.data import DataLoader
from GNN.Val_GNN import Val_GNN
from Data_Processing.collate_fn_residue import collate_fn_residue
from Data_Processing.collate_fn_amphi_59 import collate_fn_amphi_59
from Data_Processing.collate_fn_amphi_64 import collate_fn_amphi_64
from Data_Processing.collate_fn_amphi_77 import collate_fn_amphi_77
from Data_Processing.collate_fn_amphi_100 import collate_fn_amphi_100
from Data_Processing.collate_fn_atom import collate_fn_atom
import pandas as pd


def Gen_GNN_Model(data_path,model_path,params):
    GNNmode = params['type']
    mode = int(params['mode'])
    if GNNmode == 1:
        model = GNN_Model_res(params)
    elif GNNmode == 2:
        model = GNN_Model_atom(params)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = initialize_model(model, device)      
    all_set = os.listdir(data_path)
   
    test_list = [filename[:-4] for filename in all_set]
    checkpoint1 = torch.load(model_path)
    model.load_state_dict(checkpoint1['state_dict'])
    #state_dict = torch.load(model_path)
    #addprefix = 'module.'
    #state_dict = {addprefix+k for k,v in state_dict.items()}
    #model.load_state_dict(state_dict)
    ss3_csv = params['ss3_csv']
    
    file_seq_map = {}
    df_map = pd.read_csv(ss3_csv+'.csv')
    for index, row in df_map.iterrows():
        file_seq_map[row['file'][:-4]] = row['aa_seq']

    
    prediction_str = "Sequence, Prediction"
    for eachsample in test_list:
        newlist = []
        newlist.append(eachsample)
        test_dataset = Dockground_Dataset(data_path, newlist)
        if GNNmode==1:
            if mode == 59:
                test_dataloader = DataLoader(test_dataset, 1,
                                        shuffle=False, num_workers=params['num_workers'],
                                        drop_last=False,collate_fn=collate_fn_amphi_59)
            elif mode == 64:
                test_dataloader = DataLoader(test_dataset, 1,
                                        shuffle=False, num_workers=params['num_workers'],
                                        drop_last=False,collate_fn=collate_fn_amphi_64)
            elif mode == 77:
                test_dataloader = DataLoader(test_dataset, 1,
                                        shuffle=False, num_workers=params['num_workers'],
                                        drop_last=False,collate_fn=collate_fn_amphi_77)
            elif mode == 100:
                test_dataloader = DataLoader(test_dataset, 1,
                                        shuffle=False, num_workers=params['num_workers'],
                                        drop_last=False,collate_fn=collate_fn_amphi_100)
            else:
                test_dataloader = DataLoader(test_dataset, 1,
                                        shuffle=False, num_workers=params['num_workers'],
                                        drop_last=False,collate_fn=collate_fn_residue)

            prediction = Val_GNN(model, test_dataloader, device, params) 
            prediction_str += "\n"+file_seq_map[eachsample] + "," + str(prediction)
        elif GNNmode==2:
            test_dataloader = DataLoader(test_dataset, 1,
                                        shuffle=False, num_workers=params['num_workers'],
                                        drop_last=False,collate_fn=collate_fn_atom)

            prediction = Val_GNN(model, test_dataloader, device, params) 
            prediction_str += "\n"+file_seq_map[eachsample] + "," + str(prediction)
    
    f1 = open('predictions.txt','w')
    f1.write(prediction_str) 
    f1.close()
            
        