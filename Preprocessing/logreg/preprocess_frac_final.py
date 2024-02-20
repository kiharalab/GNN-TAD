import os
import numpy as np
import pandas as pd
import sys
from tqdm import tqdm
import pickle

def feature_dump_exactcount(csvpath,output_path):
    #df = pd.read_csv(sys.argv[1]) #path of csv file containing SS

    amino_id = {"A":1,"R":2,"N":3,"D":4,"C":5,"Q":6,"E":7,"G":8,"H":9,"I":10,"L":11,"K":12,"M":13,"F":14,"P":15,"S":16,"T":17,"W":18,"Y":19,"V":20}
    
    seqlen = 30
    
    try:
        os.mkdir(output_path)
    except:
        pass
    
    allcsvs = [csvpath]
    filesize = 0
    for eachcsv in allcsvs:
        df = pd.read_csv(eachcsv)
        filesize += len(df)
    #filesize = len(df)
    #print(df)
    #return
    final_features = np.zeros((filesize,21))
    iter_files = 0
    for eachcsv in tqdm(allcsvs):
        df = pd.read_csv(eachcsv)
        for index,row in tqdm(df.iterrows()):
            rowf = np.zeros(21)
            filenm = row['file']
            
            #print(filenm)
            aaseq1 = row['aa_seq']
            ss81 = row['ss8']
            aaseq = ''
            ss8 = ''
            if len(aaseq1) != seqlen:
                f1 = open('exclamations_frac.txt','a')
                f1.write(str(filenm) + " : " + str(aaseq) + "\n")
                f1.close()
                for i in range(len(aaseq1)):
                    if aaseq1[i] != '!':
                        aaseq += aaseq1[i]
                        ss8 += ss81[i]
                        

            else:
                aaseq = aaseq1
                ss8 = ss81
            #print(aaseq)
            for i in range(len(aaseq)):
                pos = amino_id[aaseq[i]] - 1 
                #print(pos)
                rowf[pos+1] += 1
            
            rowf[0] = int(filenm[:-4])
            #print(rowf)
            final_features[iter_files] = rowf
            iter_files+=1

            #prep_file=os.path.join(output_path,filenm[:-4])
            #np.savez(prep_file,  H=rowf, Y=label)
    column_names = ['curfile','A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
    dframe = pd.DataFrame(final_features,columns = column_names)

    #print(dframe)
    pickle_file = open(output_path + '/preprocessed_exact_count.p','wb')
    pickle.dump(dframe,pickle_file)
    pickle_file.close()

        
        

