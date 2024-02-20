import os
import numpy as np
from tqdm import tqdm
import sys

def build_plddt(filename,numcoords):
    Structure = open(filename, 'r')
    itrres = 0
    plddt_bins = np.zeros((numcoords,7)) #<40,40-50,50-60...90-100 etc.

    for line in Structure:
        #print(line)
        try:
            line = line.split()
            if(line[0]!="ATOM"):
                continue
            calpha = str(line[2]).strip()
            
            if (calpha == 'CA'):
                plddt = float(str(line[10]).strip())
                #plddt_bins
                if(plddt < 40):
                    plddt_bins[itrres][0] = 1
                elif(plddt >=40 and plddt < 50):
                    plddt_bins[itrres][1] = 1
                elif(plddt >= 50 and plddt < 60):
                    plddt_bins[itrres][2] = 1
                elif(plddt >= 60 and plddt < 70):
                    plddt_bins[itrres][3] = 1
                elif (plddt >= 70 and plddt < 80):
                    plddt_bins[itrres][4] = 1
                elif (plddt >= 80 and plddt < 90):
                    plddt_bins[itrres][5] = 1
                elif (plddt >=90 and plddt <= 100):
                    plddt_bins[itrres][6] = 1
                
                itrres += 1
                              
        except:
            pass
    
    
    return plddt_bins


def combine_features():
    input_path = sys.argv[1] #path to the pdb files
    preprocessed_old = sys.argv[2] #old preprocessed features which we will add to
    output_path = sys.argv[3]
    inputfiles = os.listdir(input_path)
    seqlen = 30
    try:
        os.mkdir(output_path)
    except:
        pass
    for eachfile in tqdm(inputfiles):
        filenm = os.path.join(input_path,eachfile)
        #print(filenm)
        onehot_plddt = build_plddt(filenm,seqlen)
        load_prep = np.load(os.path.join(preprocessed_old,eachfile[:-4] + ".npz"))
        prev_features = load_prep['H']
        features = np.concatenate((prev_features,onehot_plddt),axis = 1)
        #print(features)
        prep_file=os.path.join(output_path,eachfile[:-4]+".npz")
        np.savez(prep_file,  H=features, A1=load_prep['A1'], A2=load_prep['A2'] ,Y=load_prep['Y'])
        
        #break
    #fig, ax = plt.subplots(figsize =(10, 7))
    #n,bins,patches = ax.hist(relACC_vals, bins = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    #plt.savefig('ASA.png')
    
    #plt.close()
    #print(df)
    #df.to_csv(outputfile+'.csv')

#combine_features()

def feature_dump_plddt(input_path,output_path):
    inputfiles = os.listdir(input_path)
    seqlen = 30


    try:
        os.mkdir(output_path)
    except:
        pass
    for eachfile in tqdm(inputfiles):
        #print(eachfile)
        filenm = os.path.join(input_path,eachfile)
        onehot_plddt = build_plddt(filenm,seqlen)
        
        prep_file=os.path.join(output_path,eachfile[:-4]+".npz")
        np.savez(prep_file,  plddt=onehot_plddt)
        
        #break


