import os
import numpy as np
import pandas as pd
import sys
from tqdm import tqdm
import pickle
import math


def get_coords(filename):
    Structure = open(filename, 'r')
    #residue_id = {"GLY": 0,"ALA": 1,"VAL": 2,"LEU": 3,"ILE": 4,"PRO": 5,"PHE": 6,"TYR": 7,"TRP": 8,"SER": 9,"THR": 10,"CYS": 11,"MET": 12,"ASN": 13,"GLN": 14,"ASP": 15,"GLU": 16,"LYS": 17,"ARG": 18,"HIS": 19}
    itrres = 0

    
    pos_coord = [] #for getting atomic coordinates


    for line in Structure:
        #print(line)
        try:
            line = line.split()
            if(line[0]!="ATOM"):
                continue
            atomname = str(line[2]).strip()
            amino = str(line[3]).strip()
            plddt = float(str(line[10]).strip())
            
            x = float(line[6])
            y = float(line[7])
            z = float(line[8])
            pos_coord.append([x,y,z])
  
                              
        except:
            pass
    
    #features = np.concatenate((onehotpos,onehotres,acidbasearoma,plddt_bins),axis = 1) 
    return pos_coord

def ROG(coords):
    center = [0,0,0]
    center = np.array(center)
    coords = np.array(coords)
    center = np.mean(coords,axis=0)


    distarry = []
    totaldiff = 0
    for eachcoord in coords:
        diffx = (eachcoord[0] - center[0])**2
        diffy = (eachcoord[1] - center[1])**2
        diffz = (eachcoord[2] - center[2])**2
        #distarry.append([diffx,diffy,diffz])
        totaldiff += diffx+diffy+diffz
    
    totaldiff = totaldiff/len(coords)
    final_rog = math.sqrt(totaldiff)
    return final_rog

        




def feature_dump_radius(csvpath,output_path,filepath):
    #df = pd.read_csv(sys.argv[1]) #path of csv file containing SS
    #csvpath = sys.argv[1]
    #output_path = sys.argv[3]
    #filepath = sys.argv[4]
    amino_id = {"A":1,"R":2,"N":3,"D":4,"C":5,"Q":6,"E":7,"G":8,"H":9,"I":10,"L":11,"K":12,"M":13,"F":14,"P":15,"S":16,"T":17,"W":18,"Y":19,"V":20}
    
    seqlen = 30
    
    try:
        os.mkdir(output_path)
    except:
        pass
    #allcsvs = ['aaseq_ss_ds1.csv','aaseq_ss_ds2.csv','aaseq_ss_ds3.csv','aaseq_ss_ds4.csv','aaseq_ss_ds5.csv','aaseq_ss_ds6.csv','aaseq_ss_ds7.csv']
    
    allcsvs = [csvpath]
    filesize = 0
    numofcols = 6
    for eachcsv in allcsvs:
        df = pd.read_csv(eachcsv)
        filesize += len(df)
    #filesize = len(df)
    #print(df)
    #return
    final_features = np.zeros((filesize,numofcols))
    iter_files = 0
    for eachcsv in tqdm(allcsvs):
        df = pd.read_csv( eachcsv)
        
        for index,row in tqdm(df.iterrows()):
            rowf = np.zeros(numofcols)
            filenm = row['file']
            coords = get_coords(filepath + '/' + filenm)
            rog = ROG(coords)
            
            
            
            rowf[0] = int(filenm[:-4])
            #print(rowf)
            if (rog >=5 and rog <=10):
                rowf[1] = 1
            elif (rog >=10 and rog <=15):
                rowf[2] = 1
            elif (rog >=15 and rog <=20):
                rowf[3] = 1
            elif (rog >=20 and rog <=25):
                rowf[4] = 1
            elif (rog >=25 and rog <=30):
                rowf[5] = 1
            #rowf[1] = rog
            final_features[iter_files] = rowf
            iter_files+=1

            #prep_file=os.path.join(output_path,filenm[:-4])
            #np.savez(prep_file,  H=rowf, Y=label)
    column_names = ['curfile','R5-10','R10-15','R15-20','R20-25','R25-30']
    dframe = pd.DataFrame(final_features,columns = column_names)

    #print(dframe)
    pickle_file = open(output_path + '/preprocessed_radius.p','wb')
    pickle.dump(dframe,pickle_file)
    pickle_file.close()

        
        

