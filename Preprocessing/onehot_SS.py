import os
import numpy as np
import math
import pandas as pd
import sys
from tqdm import tqdm

def build_ss(filename,ss,numcoords,seq):
    ss_id = {'A': 0,'C': 1,'B': 2}
    itrres = 0
    passed = 0

    onehot_ss = np.zeros((numcoords,3)) #numcoords is 30 for this dataset
    #print(ss)
    #print(len(ss))
    if len(ss)!=30:
        if '!' not in seq:
            f = open('errorfile.txt','a')
            f.write(str(filename) + " : " + str(seq) + "\n")
            f.close()
            passed = 1
        else:
            f1 = open('exclamations.txt','a')
            f1.write(str(filename) + " : " + str(seq) + "\n")
            f1.close()
            newstruct = ''
            for i in range(len(seq)):
                if seq[i] != '!':
                    newstruct += ss[i]
            for i in range(0,len(newstruct)):
                #print(ss[i])
                onehot_ss[i][ss_id[newstruct[i]]] = 1
    else:
        for i in range(0,len(ss)):
            #print(ss[i])
            onehot_ss[i][ss_id[ss[i]]] = 1
    
    
    return onehot_ss,passed

def combine_features_SS(csvpath,preprocessed_53,output_path):
    df = pd.read_csv(csvpath) #path of csv file containing SS
    seqlen = 30
    
    try:
        os.mkdir(output_path)
    except:
        pass

    for index,row in tqdm(df.iterrows()):
        filenm = row['file']
        #print(filenm)
        ss3 = row['ss3']
        seq = row['aa_seq']
        onehot_ss,passed = build_ss(filenm,ss3,seqlen,seq)
        #print(filenm)
        #print(onehot_ss)
        if passed == 1:
            continue
        
        
        load_prep = np.load(os.path.join(preprocessed_53,filenm[:-4] + ".npz"))
        prev_features = load_prep['H']
        features = np.concatenate((prev_features,onehot_ss),axis = 1)
        #print(features)
        prep_file=os.path.join(output_path,filenm[:-4]+".npz")
        np.savez(prep_file,  H=features, A1=load_prep['A1'], A2=load_prep['A2'])
        #break
        

