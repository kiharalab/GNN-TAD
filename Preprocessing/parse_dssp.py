import os
import numpy as np
import sys
import pandas as pd 
from tqdm import tqdm

def read_dssp(dssp_dir,dssp_file):
    filenam = dssp_dir + '/' + dssp_file
    startReading = 0
    #ss_id = []
    ss_id = ''
    amino_seq = ''
    #print(dssp_file)
    with open(filenam,'r') as f:
        for line in f.readlines():
            if startReading == 1:
                #print(line)
                struc = line[16]
                amino = line[13]
                #print("structure is: " + struc)
                #print("amino is: " + amino)
                amino_seq += amino
                if struc == ' ':
                    #print("no structure found")
                    #ss_id.append('-')
                    ss_id += '-'
                else:
                    #ss_id.append(struc)
                    ss_id += str(struc)
                #break
            if '#' in line and 'RESIDUE' in line and 'AA' in line and 'STRUCTURE' in line:
                #print(line)
                startReading = 1
    
    #print(ss_id)   
    return amino_seq,ss_id 

def ss823(ss_id_8):
    #ss_dict = {'H':0,'B':2,'E':2,'G':0,'I':0,'T':1,'S':1,'-':1,'P':1} #0 - helix, 1 - coil, 2 - beta 
    ss_id_3 = {'H':'A','B':'B','E':'B','G':'A','I':'A','T':'C','S':'C','-':'C','P':'C'}

    ss_str = ''
    for ss in ss_id_8:
        ss_str += ss_id_3[ss]

    return ss_str

def parse_dssp_files(input_path,outputfile):
    inputfiles = os.listdir(input_path)
    df = pd.DataFrame(columns=['aa_seq','ss8','ss3','file'])
    for eachfile in tqdm(inputfiles):
        amino_seq,ss_id = read_dssp(input_path,eachfile)
        ss3 = ss823(ss_id)
        df.loc[len(df.index)] = [amino_seq,ss_id,ss3,eachfile[:-5] + '.pdb']
    print(df)
    df.to_csv(outputfile+'.csv')

