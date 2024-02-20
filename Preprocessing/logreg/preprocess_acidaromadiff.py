import os
import numpy as np
import pandas as pd
import sys
from tqdm import tqdm
import pickle

def feature_dump_acidaromadiff(csvpath,output_path):
    #df = pd.read_csv(sys.argv[1]) #path of csv file containing SS
    #csvpath = sys.argv[1]
    #output_path = sys.argv[3]
    amino_id = {"A":1,"R":2,"N":3,"D":4,"C":5,"Q":6,"E":7,"G":8,"H":9,"I":10,"L":11,"K":12,"M":13,"F":14,"P":15,"S":16,"T":17,"W":18,"Y":19,"V":20}
    acid_id = {"D":0,"E":0}
    aroma_id = {'W':0,'F':0,'Y':0}
    seqlen = 30
    
    try:
        os.mkdir(output_path)
    except:
        pass
    #allcsvs = ['aaseq_ss_ds1.csv','aaseq_ss_ds2.csv','aaseq_ss_ds3.csv','aaseq_ss_ds4.csv','aaseq_ss_ds5.csv','aaseq_ss_ds6.csv','aaseq_ss_ds7.csv']
    allcsvs = [csvpath]
    filesize = 0
    for eachcsv in allcsvs:
        df = pd.read_csv(eachcsv)
        filesize += len(df)
    #filesize = len(df)
    #print(df)
    #return
    numofcols = 8
    final_features = np.zeros((filesize,numofcols))
    iter_files = 0
    for eachcsv in tqdm(allcsvs):
        df = pd.read_csv(eachcsv)
        for index,row in tqdm(df.iterrows()):
            rowf = np.zeros(numofcols)
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
                if(aaseq[i] in acid_id):
                    acid_id[aaseq[i]] += 1
                elif(aaseq[i] in aroma_id):
                    aroma_id[aaseq[i]] += 1
            
            aroma_sum = 0
            acid_sum = 0

            for aromas in aroma_id:
                aroma_sum += aroma_id[aromas]
            for acids in acid_id:
                acid_sum += acid_id[acids]
            diff = aroma_sum - acid_sum

            if diff==0:
                rowf[4] = 1.0
            elif (diff > 0 and diff <= 5):
                rowf[5] = 1.0
            elif(diff > 5 and diff <=10):
                rowf[6] = 1.0
            elif(diff > 10):
                rowf[7] = 1.0
            elif(diff >= -5 and diff < 0):
                rowf[3] = 1.0
            elif(diff >= -10 and diff < -5):
                rowf[2] = 1.0
            elif(diff < -10):
                rowf[1] = 1.0

            
            rowf[0] = int(filenm[:-4])
            #print(rowf)
            final_features[iter_files] = rowf
            iter_files+=1
            for aromas in aroma_id:
                aroma_id[aromas] = 0
            for acids in acid_id:
                acid_id[acids] = 0

            #prep_file=os.path.join(output_path,filenm[:-4])
            #np.savez(prep_file,  H=rowf, Y=label)
    column_names = ['curfile','<-10','-10 to -5','-5 to 0','0','0 to +5','+5 to +10','>+10']
    #column_names = ['curfile','A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V','label']
    
    dframe = pd.DataFrame(final_features,columns = column_names)

    #print(dframe)
    pickle_file = open(output_path + '/preprocessed_acidaromadiff.p','wb')
    pickle.dump(dframe,pickle_file)
    pickle_file.close()

        
        

