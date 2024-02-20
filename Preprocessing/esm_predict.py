import torch
import esm
import pandas as pd
import os
from tqdm import tqdm

def load_model():
    model = esm.pretrained.esmfold_v1()
    model = model.eval().cuda()
    return model

def do_prediction(df,dirname):
    itr = 0
    model = load_model()
    for index, row in tqdm(df.iterrows()):
        sequence = row['aa_seq']
        filename = row['id']

        with torch.no_grad():
            output = model.infer_pdb(sequence)

        with open(dirname + "/" + str(filename)+".pdb", "w") as f:
            f.write(output)
        
        itr += 1



def read_data(csvfile):
    df = pd.read_csv(csvfile)
    #print(df)
    df.drop(columns=df.columns[0], axis=1, inplace=True)
    return df


def predict_structure(input_path,output_path,esm_model_path):
    csvfile = input_path
    output_dir = output_path
    torch.hub.set_dir(esm_model_path)
    try:
        os.mkdir(output_dir)
    except:
        pass

    df = read_data(csvfile)
    #print(df)
    do_prediction(df,output_dir)




