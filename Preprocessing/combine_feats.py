import os
import numpy as np
import math
import pandas as pd
import sys
from tqdm import tqdm

def combine_features(input_path1,input_path2,preprocessed_prev,output_path):
    #input_path1 = sys.argv[1] #path to new feats1 = plddt
    #input_path2 = sys.argv[2] #path to new feats2 = ASA
    #preprocessed_prev = sys.argv[3] #name of folder containing preprocessed files with base feature combo = pos_res_aa_ss
    #output_path = sys.argv[4]
    seqlen = 30
    
    try:
        os.mkdir(output_path)
    except:
        pass

    allfiles = os.listdir(input_path1)
    for eachfile in tqdm(allfiles):
        feat1 = np.load(os.path.join(preprocessed_prev,eachfile))['H']
        a1 = np.load(os.path.join(preprocessed_prev,eachfile))['A1']
        a2 = np.load(os.path.join(preprocessed_prev,eachfile))['A2']
        feat2 = np.load(os.path.join(input_path1,eachfile))['plddt']
        feat3 = np.load(os.path.join(input_path2,eachfile))['asa']

        features = np.concatenate((feat1,feat2,feat3),axis = 1)
        prep_file=os.path.join(output_path,eachfile)
        np.savez(prep_file,  H=features, A1=a1, A2=a2)
        

