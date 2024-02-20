import os
import numpy as np
import pandas as pd
import sys
from tqdm import tqdm
from math import cos, sin
from scipy.integrate import quad
import pickle

FGR = 0.0174533

def cos1(angle):
	funcval = 0
	for i in range(len(hydro)):
		funcval += (hydro[i]-meanh)*cos(angle*i*FGR)

	return funcval**2

def sin1(angle):
	funcval = 0
	for i in range(len(hydro)):
		funcval += (hydro[i]-meanh)*sin(angle*i*FGR)

	return funcval**2

def total_sum_norm_integral(h, mean):
	global hydro, meanh
	hydro = h
	meanh = mean
	#Pw = cos1(h,mean,angle) + sin1(h,mean,angle)
	AI_num = (quad(cos1,85,110)[0] + quad(sin1,85,110)[0])/(25*FGR)
	AI_den = (quad(cos1,1,180)[0] + quad(sin1,1,180)[0])/(180*FGR)
	AI = AI_num/AI_den
	return AI

def calculate_amphipathic_index(seq):
    #seq = struct['seq']
    hydrophobic_table = {'A':-0.96,'R':0.75,'N':-1.94,'D':-5.68,'C':4.54,'Q':-5.3,'E':-3.86,'G':-1.28,'H':-0.62,'I':5.54,'L':6.81,'K':-5.62,'M':4.76,'F':5.06,'P':-4.47,'S':-1.92,'T':-3.99,'W':0.21,'Y':3.34,'V':5.39} #PRIFT original
    #hydrophobic_table = {'A':1.8,'R':-4.5,'N':-3.9,'D':-3.5,'C':-0.4,'Q':-3.5,'E':-3.5,'G':0,'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.45,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2} #KyleDolittle scale
    hydrophobic_scores = [
        hydrophobic_table[aa]
        for aa in seq
        if aa in hydrophobic_table
    ]
    amount = len(hydrophobic_scores)
    amount = amount if amount > 0 else 1.
    #mean = sum(hydrophobic_scores) / amount
    mean = 0
    ai2 = total_sum_norm_integral(hydrophobic_scores,mean)
    return ai2

def amphi_avg_eachseq(seq,struct,interval,ss_id,seqlen): #seq = aaseq
	numofstride = seqlen - interval + 1
	amphi_dict = {}
	for i in range(len(seq)):
		amphi_dict[seq[i] + '_' + str(i)] = [-1,0]

	#print(amphi_dict)

	for i in range(numofstride):
		#i is starting index of subset helix
		new_seq = seq[i:i+interval]
		new_struct = struct[i:i+interval]
		#print(new_struct)
		flag = 1
		for val in new_struct:
			if ss_id[val] != 0:
				flag = 0
				break
		#if all helices - check amphipathic
		if flag == 1: #all helices in new seq
			ai2=calculate_amphipathic_index(new_seq)
			for j in range(i,i+interval):
				seq_identifier = seq[j] + '_' + str(j)
				#print(seq_identifier)
				if(amphi_dict[seq_identifier][0] == -1):
					amphi_dict[seq_identifier][0] = ai2
					amphi_dict[seq_identifier][1] = 1
				else:
					amphi_dict[seq_identifier][0] += ai2
					amphi_dict[seq_identifier][1] += 1


	#print(amphi_dict)
	avgarrays = []
	for residue in amphi_dict:
		totalamphi = amphi_dict[residue][0]
		if totalamphi == -1:
			avgarrays.append(-1)
			continue
		countamphi = amphi_dict[residue][1]
		amphi_dict[residue][0] = totalamphi/countamphi
		avgarrays.append(amphi_dict[residue][0])

	#print(amphi_dict)


	return avgarrays

def build_amphi_01(aaseq,ss8,numcoords,interval):
    ss_id = {'H': 0, 'B': 2, 'E': 2, 'G': 0, 'I': 0, 'T': 1, 'S': 1, '-': 1,'P':1}
    avgarrays = amphi_avg_eachseq(aaseq,ss8,interval,ss_id,numcoords)
    #print(avgarrays)
    amphi_bins = np.zeros((numcoords,47)) #-1,0-0.2,0.2-0.4 etc.
    for i in range(len(avgarrays)):
        cur_amphi = avgarrays[i]
        if(cur_amphi == -1):
            amphi_bins[i][0] = 1
        elif(cur_amphi >= 0 and cur_amphi < 0.1):
            amphi_bins[i][1] = 1
        elif(cur_amphi >= 0.1 and cur_amphi < 0.2):
            amphi_bins[i][2] = 1
        elif(cur_amphi >= 0.2 and cur_amphi < 0.3):
            amphi_bins[i][3] = 1
        elif(cur_amphi >= 0.3 and cur_amphi < 0.4):
            amphi_bins[i][4] = 1
        elif(cur_amphi >= 0.4 and cur_amphi < 0.5):
            amphi_bins[i][5] = 1
        elif(cur_amphi >= 0.5 and cur_amphi < 0.6):
            amphi_bins[i][6] = 1
        elif(cur_amphi >= 0.6 and cur_amphi < 0.7):
            amphi_bins[i][7] = 1
        elif(cur_amphi >= 0.7 and cur_amphi < 0.8):
            amphi_bins[i][8] = 1
        elif(cur_amphi >= 0.8 and cur_amphi < 0.9):
            amphi_bins[i][9] = 1
        elif(cur_amphi >= 0.9 and cur_amphi < 1):
            amphi_bins[i][10] = 1
        elif(cur_amphi >= 1 and cur_amphi < 1.1):
            amphi_bins[i][11] = 1
        elif(cur_amphi >= 1.1 and cur_amphi < 1.2):
            amphi_bins[i][12] = 1
        elif(cur_amphi >= 1.2 and cur_amphi < 1.3):
            amphi_bins[i][13] = 1
        elif(cur_amphi >= 1.3 and cur_amphi < 1.4):
            amphi_bins[i][14] = 1
        elif(cur_amphi >= 1.4 and cur_amphi < 1.5):
            amphi_bins[i][15] = 1
        elif(cur_amphi >= 1.5 and cur_amphi < 1.6):
            amphi_bins[i][16] = 1
        elif(cur_amphi >= 1.6 and cur_amphi < 1.7):
            amphi_bins[i][17] = 1
        elif(cur_amphi >= 1.7 and cur_amphi < 1.8):
            amphi_bins[i][18] = 1
        elif(cur_amphi >= 1.8 and cur_amphi < 1.9):
            amphi_bins[i][19] = 1
        elif(cur_amphi >= 1.9 and cur_amphi < 2):
            amphi_bins[i][20] = 1
        elif(cur_amphi >= 2 and cur_amphi < 2.1):
            amphi_bins[i][21] = 1
        elif(cur_amphi >= 2.1 and cur_amphi < 2.2):
            amphi_bins[i][22] = 1
        elif(cur_amphi >= 2.2 and cur_amphi < 2.3):
            amphi_bins[i][23] = 1
        elif(cur_amphi >= 2.3 and cur_amphi < 2.4):
            amphi_bins[i][24] = 1
        elif(cur_amphi >= 2.4 and cur_amphi < 2.5):
            amphi_bins[i][25] = 1
        elif(cur_amphi >= 2.5 and cur_amphi < 2.6):
            amphi_bins[i][26] = 1
        elif(cur_amphi >= 2.6 and cur_amphi < 2.7):
            amphi_bins[i][27] = 1
        elif(cur_amphi >= 2.7 and cur_amphi < 2.8):
            amphi_bins[i][28] = 1
        elif(cur_amphi >= 2.8 and cur_amphi < 2.9):
            amphi_bins[i][29] = 1
        elif(cur_amphi >= 2.9 and cur_amphi < 3):
            amphi_bins[i][30] = 1
        elif(cur_amphi >= 3 and cur_amphi < 3.1):
            amphi_bins[i][31] = 1
        elif(cur_amphi >= 3.1 and cur_amphi < 3.2):
            amphi_bins[i][32] = 1
        elif(cur_amphi >= 3.2 and cur_amphi < 3.3):
            amphi_bins[i][33] = 1
        elif(cur_amphi >= 3.3 and cur_amphi < 3.4):
            amphi_bins[i][34] = 1
        elif(cur_amphi >= 3.4 and cur_amphi < 3.5):
            amphi_bins[i][35] = 1
        elif(cur_amphi >= 3.5 and cur_amphi < 3.6):
            amphi_bins[i][36] = 1
        elif(cur_amphi >= 3.6 and cur_amphi < 3.7):
            amphi_bins[i][37] = 1
        elif(cur_amphi >= 3.7 and cur_amphi < 3.8):
            amphi_bins[i][38] = 1
        elif(cur_amphi >= 3.8 and cur_amphi < 3.9):
            amphi_bins[i][39] = 1
        elif(cur_amphi >= 3.9 and cur_amphi < 4):
            amphi_bins[i][40] = 1
        elif(cur_amphi >= 4 and cur_amphi < 4.1):
            amphi_bins[i][41] = 1
        elif(cur_amphi >= 4.1 and cur_amphi < 4.2):
            amphi_bins[i][42] = 1
        elif(cur_amphi >= 4.2 and cur_amphi < 4.3):
            amphi_bins[i][43] = 1
        elif(cur_amphi >= 4.3 and cur_amphi < 4.4):
            amphi_bins[i][44] = 1
        elif(cur_amphi >= 4.4 and cur_amphi < 4.5):
            amphi_bins[i][45] = 1
        elif(cur_amphi >= 4.5 and cur_amphi <= 4.6):
            amphi_bins[i][46] = 1
    return amphi_bins


def build_amphi_02(aaseq,ss8,numcoords,interval):
    ss_id = {'H': 0, 'B': 2, 'E': 2, 'G': 0, 'I': 0, 'T': 1, 'S': 1, '-': 1,'P':1}
    avgarrays = amphi_avg_eachseq(aaseq,ss8,interval,ss_id,numcoords)
    #print(avgarrays)
    amphi_bins = np.zeros((numcoords,24)) #-1,0-0.2,0.2-0.4 etc.
    for i in range(len(avgarrays)):
        cur_amphi = avgarrays[i]
        if(cur_amphi == -1):
            amphi_bins[i][0] = 1
        elif(cur_amphi >= 0 and cur_amphi < 0.2):
            amphi_bins[i][1] = 1
        elif(cur_amphi >= 0.2 and cur_amphi < 0.4):
            amphi_bins[i][2] = 1
        elif(cur_amphi >= 0.4 and cur_amphi < 0.6):
            amphi_bins[i][3] = 1
        elif(cur_amphi >= 0.6 and cur_amphi < 0.8):
            amphi_bins[i][4] = 1
        elif(cur_amphi >= 0.8 and cur_amphi < 1):
            amphi_bins[i][5] = 1
        elif(cur_amphi >= 1 and cur_amphi < 1.2):
            amphi_bins[i][6] = 1
        elif(cur_amphi >= 1.2 and cur_amphi < 1.4):
            amphi_bins[i][7] = 1
        elif(cur_amphi >= 1.4 and cur_amphi < 1.6):
            amphi_bins[i][8] = 1
        elif(cur_amphi >= 1.6 and cur_amphi < 1.8):
            amphi_bins[i][9] = 1
        elif(cur_amphi >= 1.8 and cur_amphi < 2):
            amphi_bins[i][10] = 1
        elif(cur_amphi >= 2 and cur_amphi < 2.2):
            amphi_bins[i][11] = 1
        elif(cur_amphi >= 2.2 and cur_amphi < 2.4):
            amphi_bins[i][12] = 1
        elif(cur_amphi >= 2.4 and cur_amphi < 2.6):
            amphi_bins[i][13] = 1
        elif(cur_amphi >= 2.6 and cur_amphi < 2.8):
            amphi_bins[i][14] = 1
        elif(cur_amphi >= 2.8 and cur_amphi < 3):
            amphi_bins[i][15] = 1
        elif(cur_amphi >= 3 and cur_amphi < 3.2):
            amphi_bins[i][16] = 1
        elif(cur_amphi >= 3.2 and cur_amphi < 3.4):
            amphi_bins[i][17] = 1
        elif(cur_amphi >= 3.4 and cur_amphi < 3.6):
            amphi_bins[i][18] = 1
        elif(cur_amphi >= 3.6 and cur_amphi < 3.8):
            amphi_bins[i][19] = 1
        elif(cur_amphi >= 3.8 and cur_amphi < 4):
            amphi_bins[i][20] = 1
        elif(cur_amphi >= 4 and cur_amphi < 4.2):
            amphi_bins[i][21] = 1
        elif(cur_amphi >= 4.2 and cur_amphi < 4.4):
            amphi_bins[i][22] = 1
        elif(cur_amphi >= 4.4 and cur_amphi <= 4.6):
            amphi_bins[i][23] = 1
    return amphi_bins

def build_amphi_05(aaseq,ss8,numcoords,interval):
    ss_id = {'H': 0, 'B': 2, 'E': 2, 'G': 0, 'I': 0, 'T': 1, 'S': 1, '-': 1,'P':1}
    avgarrays = amphi_avg_eachseq(aaseq,ss8,interval,ss_id,numcoords)
    #print(avgarrays)
    amphi_bins = np.zeros((numcoords,11)) #-1,0-0.2,0.2-0.4 etc.
    for i in range(len(avgarrays)):
        cur_amphi = avgarrays[i]
        if(cur_amphi == -1):
            amphi_bins[i][0] = 1
        elif(cur_amphi >= 0 and cur_amphi < 0.5):
            amphi_bins[i][1] = 1
        elif(cur_amphi >= 0.5 and cur_amphi < 1):
            amphi_bins[i][2] = 1
        elif(cur_amphi >= 1 and cur_amphi < 1.5):
            amphi_bins[i][3] = 1
        elif(cur_amphi >= 1.5 and cur_amphi < 2):
            amphi_bins[i][4] = 1
        elif(cur_amphi >= 2 and cur_amphi < 2.5):
            amphi_bins[i][5] = 1
        elif(cur_amphi >= 2.5 and cur_amphi < 3):
            amphi_bins[i][6] = 1
        elif(cur_amphi >= 3 and cur_amphi < 3.5):
            amphi_bins[i][7] = 1
        elif(cur_amphi >= 3.5 and cur_amphi < 4):
            amphi_bins[i][8] = 1
        elif(cur_amphi >= 4 and cur_amphi < 4.5):
            amphi_bins[i][9] = 1
        elif(cur_amphi >= 4.5 and cur_amphi <= 5):
            amphi_bins[i][10] = 1
        
    return amphi_bins



def build_amphi_1(aaseq,ss8,numcoords,interval):
    ss_id = {'H': 0, 'B': 2, 'E': 2, 'G': 0, 'I': 0, 'T': 1, 'S': 1, '-': 1,'P':1}
    avgarrays = amphi_avg_eachseq(aaseq,ss8,interval,ss_id,numcoords)
    #print(avgarrays)
    amphi_bins = np.zeros((numcoords,6)) #-1,0-0.2,0.2-0.4 etc.
    for i in range(len(avgarrays)):
        cur_amphi = avgarrays[i]
        if(cur_amphi == -1):
            amphi_bins[i][0] = 1
        elif(cur_amphi >= 0 and cur_amphi < 1):
            amphi_bins[i][1] = 1
        elif(cur_amphi >= 1 and cur_amphi < 2):
            amphi_bins[i][2] = 1
        elif(cur_amphi >= 2 and cur_amphi < 3):
            amphi_bins[i][3] = 1
        elif(cur_amphi >= 3 and cur_amphi < 4):
            amphi_bins[i][4] = 1
        elif(cur_amphi >= 4 and cur_amphi < 5):
            amphi_bins[i][5] = 1
    return amphi_bins


def feature_dump_amphi(csvpath,output_path,mode):
    df = pd.read_csv(csvpath) #path of csv file containing SS
    #output_path = sys.argv[2]
    #files_filtered = sys.argv[3]
    #df3 = open(files_filtered,'rb')
    #newarr = pickle.load(df3)
    #df3.close()
    #print(newarr)
    #return
    seqlen = 30
    interval = 10
    
    try:
        os.mkdir(output_path)
    except:
        pass

    for index,row in tqdm(df.iterrows()):
        filenm = row['file']
        #if filenm not in newarr:
        #     continue
        #print(filenm)
        aaseq1 = row['aa_seq']
        ss81 = row['ss8']
        aaseq = ''
        ss8 = ''
        if len(aaseq1) != seqlen:
            f1 = open('exclamations_amphi.txt','a')
            f1.write(str(filenm) + " : " + str(aaseq) + "\n")
            f1.close()
            for i in range(len(aaseq1)):
                if aaseq1[i] != '!':
                    aaseq += aaseq1[i]
                    ss8 += ss81[i]
        else:
            aaseq = aaseq1
            ss8 = ss81

        
        prep_file=os.path.join(output_path,filenm[:-4]+".npz")
        if mode == 59:
            onehot_amphi_bin1 = build_amphi_1(aaseq,ss8,seqlen,interval)
            np.savez(prep_file,  amphi=onehot_amphi_bin1)
        elif mode == 64:
            onehot_amphi_bin05 = build_amphi_05(aaseq,ss8,seqlen,interval)
            np.savez(prep_file,  amphi=onehot_amphi_bin05)
        elif mode == 77:
            onehot_amphi_bin02 = build_amphi_02(aaseq,ss8,seqlen,interval)
            np.savez(prep_file,  amphi=onehot_amphi_bin02)
        elif mode == 100:
            onehot_amphi_bin01 = build_amphi_01(aaseq,ss8,seqlen,interval)
            np.savez(prep_file,  amphi=onehot_amphi_bin01)
        

