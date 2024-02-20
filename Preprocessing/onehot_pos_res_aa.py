import os
import numpy as np
import math

def build_pos_res_aa_plddt(filename,numcoords):
    Structure = open(filename, 'r')
    residue_id = {"GLY": 0,"ALA": 1,"VAL": 2,"LEU": 3,"ILE": 4,"PRO": 5,"PHE": 6,"TYR": 7,"TRP": 8,"SER": 9,"THR": 10,"CYS": 11,"MET": 12,"ASN": 13,"GLN": 14,"ASP": 15,"GLU": 16,"LYS": 17,"ARG": 18,"HIS": 19}
    itrres = 0

    onehotpos = np.zeros((numcoords,numcoords)) #numcoords is 30 for this dataset
    onehotres = np.zeros((numcoords,len(residue_id))) #20 residues
    acidbasearoma = np.zeros((numcoords,3))
    plddt_bins = np.zeros((numcoords,10)) #0-10,10-20...90-100 etc.

    acids = ['ASP','GLU']
    bases = ['ARG','HIS','LYS']
    #bases = ['ARG','LYS']
    aromas = ['TRP','PHE','TYR']
    pos_coord = [] #for getting atomic coordinates



    for line in Structure:
        #print(line)
        try:
            line = line.split()
            if(line[0]!="ATOM"):
                continue
            calpha = str(line[2]).strip()
            amino = str(line[3]).strip()
            plddt = float(str(line[10]).strip())
            

            
            if (calpha == 'CA'):
                #print(str(itrres) + ":" + str(amino) + ":" + str(plddt))
                x = float(line[6])
                y = float(line[7])
                z = float(line[8])
                pos_coord.append([x,y,z])
                
                onehotpos[itrres][itrres] = 1
                onehotres[itrres][residue_id[amino]] = 1

                if(amino in acids):
                    acidbasearoma[itrres][0] = 1
                elif(amino in bases):
                    acidbasearoma[itrres][1] = 1
                elif(amino in aromas):
                    acidbasearoma[itrres][2] = 1
                
                itrres+=1
                continue              
        except:
            pass
    
    #features = np.concatenate((onehotpos,onehotres,acidbasearoma,plddt_bins),axis = 1) 
    features = np.concatenate((onehotpos,onehotres,acidbasearoma),axis = 1) 
    '''
    Feature set description for this function:
    1 to 30 = one hot pos
    31 to 50 = one hot res
    51 to 53 = acid base aroma
    '''

    return features,pos_coord