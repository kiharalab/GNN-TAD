import os
import numpy as np
from tqdm import tqdm

def read_dssp(dssp_dir,dssp_file):
    maxACC = {
        "A": 106.0,
        "R": 248.0,
        "N": 157.0,
        "D": 163.0,
        "C": 135.0,
        "Q": 198.0,
        "E": 194.0,
        "G": 84.0,
        "H": 184.0,
        "I": 169.0,
        "L": 164.0,
        "K": 205.0,
        "M": 188.0,
        "F": 197.0,
        "P": 136.0,
        "S": 130.0,
        "T": 142.0,
        "W": 227.0,
        "Y": 222.0,
        "V": 142.0,
    }
    filenam = dssp_dir + '/' + dssp_file
    startReading = 0
    #ss_id = []
    ss_id = ''
    amino_seq = ''
    #print(dssp_file)
    relAcc = []
    #tempacc = []
    with open(filenam,'r') as f:
        for line in f.readlines():
            if startReading == 1:
                #print(line)
                struc = line[16]
                amino = line[13]
                if amino == '!':
                    continue
                acc = int(line[35:38])
                
                relAcc_val = round(acc/maxACC[str(amino)],3)
                if relAcc_val > 1.0:
                    relAcc_val = 1.0
                relAcc.append(relAcc_val)
                #tempacc.append(acc)
                
                
            if '#' in line and 'RESIDUE' in line and 'AA' in line and 'ACC' in line:
                #print(line)
                startReading = 1
    #print(tempacc)
    #print(relAcc)
    return relAcc

def build_ASA(filename,relAcc,numcoords):
    asa_bins = np.zeros((numcoords,10))
    for i in range(len(relAcc)):
        cur_asa = relAcc[i]
        if(cur_asa >= 0 and cur_asa < 0.1):
            asa_bins[i][0] = 1
        elif(cur_asa >= 0.1 and cur_asa < 0.2):
            asa_bins[i][1] = 1
        elif(cur_asa >= 0.2 and cur_asa < 0.3):
            asa_bins[i][2] = 1
        elif(cur_asa >= 0.3 and cur_asa < 0.4):
            asa_bins[i][3] = 1
        elif(cur_asa >= 0.4 and cur_asa < 0.5):
            asa_bins[i][4] = 1
        elif(cur_asa >= 0.5 and cur_asa < 0.6):
            asa_bins[i][5] = 1
        elif(cur_asa >= 0.6 and cur_asa < 0.7):
            asa_bins[i][6] = 1
        elif(cur_asa >= 0.7 and cur_asa < 0.8):
            asa_bins[i][7] = 1
        elif(cur_asa >= 0.8 and cur_asa < 0.9):
            asa_bins[i][8] = 1
        elif(cur_asa >= 0.9 and cur_asa <= 1):
            asa_bins[i][9] = 1
    return asa_bins



def feature_dump_asa(input_path,output_path):
    #input_path = sys.argv[1] #path to the dssp files
    inputfiles = os.listdir(input_path)
    seqlen = 30
    #df = pd.DataFrame(columns=['aa_seq','ss8','ss3','file'])
    try:
        os.mkdir(output_path)
    except:
        pass
    for eachfile in tqdm(inputfiles):
        #print(eachfile)
        relAcc = read_dssp(input_path,eachfile)
        onehot_ASA = build_ASA(eachfile,relAcc,seqlen)
        
        prep_file=os.path.join(output_path,eachfile[:-5]+".npz")
        np.savez(prep_file,  asa=onehot_ASA)
        
        #break
    
