import os
import numpy as np
import pandas as pd
import sys
from tqdm import tqdm
import pickle
import math


def get_int_coords(filename,acidbasearoma):
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
            
            if ((amino in acidbasearoma) and (atomname=='CA')):
                x = float(line[6])
                y = float(line[7])
                z = float(line[8])
                pos_coord.append([x,y,z])
  
                              
        except:
            pass
    
    #features = np.concatenate((onehotpos,onehotres,acidbasearoma,plddt_bins),axis = 1) 
    return pos_coord

def find_diff_dist(aroma_coord,acid_coord,numofcol):
    rowf = np.zeros(numofcol)
    if (len(aroma_coord) >= 1 and len(acid_coord) >= 1):
        for i in range(0,len(aroma_coord)):
            #print(aroma_coord[i])
            x1 = aroma_coord[i][0]
            y1 = aroma_coord[i][1]
            z1 = aroma_coord[i][2]
            for j in range(0,len(acid_coord)):
                x2 = acid_coord[j][0]
                y2 = acid_coord[j][1]
                z2 = acid_coord[j][2]
                #print("x1,y1,z1: " +  str(x1) + "," + str(y1) + "," + str(z1))
                #print("x2,y2,z2: " +  str(x2) + "," + str(y2) + "," + str(z2))

        
                dist = (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2)
                dist = math.sqrt(dist)
                #print(dist)

                reduced_dist = int(dist/10)
                #print(reduced_dist)
                if(reduced_dist < 10):
                    rowf[reduced_dist+1] += 1
                else:
                    rowf[11] += 1 
                #checking_high.append(dist) 
                #print(rowf)
    else:
        rowf[12] += 1
  

    return rowf



        




def feature_dump_acidbasedist(csvpath,output_path,filepath):
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
    numofcols = 13
    for eachcsv in allcsvs:
        df = pd.read_csv(eachcsv)
        filesize += len(df)
    #filesize = len(df)
    #print(df)
    #return
    final_features = np.zeros((filesize,numofcols))
    iter_files = 0
    acids = ['ASP','GLU']
    #aromas = ['TYR','TRP','PHE']
    aromas = ['ARG','HIS','LYS']
    for eachcsv in tqdm(allcsvs):
        df = pd.read_csv(eachcsv)
        
        for index,row in tqdm(df.iterrows()):
            rowf = np.zeros(numofcols)
            filenm = row['file']
            coords_acids = get_int_coords(filepath + '/' + filenm,acids)
            coords_aromas = get_int_coords(filepath + '/' + filenm,aromas)
            rowf = None
            #find acidic distances by histogram 
      
            rowf = find_diff_dist(coords_aromas,coords_acids,numofcols)
            
            
            
            rowf[0] = int(filenm[:-4])
            #print(rowf)
            
            final_features[iter_files] = rowf
            iter_files+=1

            #prep_file=os.path.join(output_path,filenm[:-4])
            #np.savez(prep_file,  H=rowf, Y=label)
    column_names = ['curfile','0 to 10','10 to 20','20 to 30','30 to 40','40 to 50','50 to 60','60 to 70','70 to 80','80 to 90','90 to 100','100+','1 or both absent']
    dframe = pd.DataFrame(final_features,columns = column_names)

    #print(dframe)
    pickle_file = open(output_path + '/preprocessed_acidbasedist.p','wb')
    pickle.dump(dframe,pickle_file)
    pickle_file.close()

        
        

