import os
import numpy as np 
import sys
from Preprocessing.onehot_pos_res_aa import build_pos_res_aa_plddt
import pickle
from tqdm import tqdm
import math

def find_dist(coords1,coords2):
    x1 = coords1[0]
    y1 = coords1[1]
    z1 = coords1[2]
    
    x2 = coords2[0]
    y2 = coords2[1]
    z2 = coords2[2]
    #print("x1,y1,z1: " +  str(x1) + "," + str(y1) + "," + str(z1))
    #print("x2,y2,z2: " +  str(x2) + "," + str(y2) + "," + str(z2))

    
    dist = (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2)
    dist = math.sqrt(dist)
    return dist

def build_matrix2(coords):
    numofres = len(coords)
    adj = np.zeros((numofres,numofres))
    for i in range(0,numofres):
        for j in range(i,numofres):
            dist = find_dist(coords[i],coords[j])
            adj[i][j] = dist
            adj[j][i] = dist

    return adj

def build_matrix1(coords):
    numofres = len(coords)
    adj = np.zeros((numofres,numofres))
    for i in range(0,numofres):
        if(i==0):
            adj[i][1] = 1

        elif(i==numofres-1):
            adj[i][numofres-2] = 1
        else:
            adj[i][i-1] = 1
            adj[i][i+1] = 1
        adj[i][i] = 1

    #print(adj)
    return adj


def build_features_pdb(input_filename,seq_len):
    onehot_pos_res_aa_plddt_feats,pos_coords = build_pos_res_aa_plddt(input_filename,seq_len)
    #print(pos_coords)
    return onehot_pos_res_aa_plddt_feats,pos_coords

def prep_input(input_filename,output_path,seq_len):
    H,coords = build_features_pdb(input_filename,seq_len)
    adj1 = build_matrix1(coords)
    adj2 = build_matrix2(coords)
    id = os.path.split(input_filename)[1].split(".")[0]
    prep_file=os.path.join(output_path,id+".npz")
    np.savez(prep_file,  H=H, A1=adj1, A2=adj2)

def preprocess_begin(input_path,output_path):
    seq_len = 30
    try:
        os.mkdir(output_path)
    except:
        pass
    paths = os.listdir(input_path)

    for path in tqdm(paths):
        path = os.path.join(input_path,path)
        prep_input(path,output_path,seq_len)


    
            

