import pandas as pd
import numpy as np
import sys
import os
import pickle

def preprocess_logreg(directoryn):
  #column 0: curfile
  #columns 1 to 117: sequence based features
  #columns = 1 to 20 -  amino acids; 21 to 50 - acid positions; 51 to 80 - base positions;
  #81 to 110 - aroma positions; 111 to 117 - acid aroma difference; 
  #
  #columns 118 to 125:structure based features
  # columns 118 to 122: radius
  # columns 123 to 125: SS3
  #
  #columns 126 to 200: distance based features
  # columns 126 to 138: acid distance
  #columns 139 to 151: aroma distance
  #columns 152 to 164: base distance

  #columns 165 to 176: acid aroma distance
  #columns 177 to 188: acid base distance
  #columns 189 to 200: aroma base distance

  # 201 - label
  
  num_features = 13

  #frac of amino
  df = pd.read_pickle(directoryn + '/preprocessed_exact_count' + '.p')
  
  #acid position
  df2 = pd.read_pickle(directoryn + '/preprocessed_acidpos' + '.p')
  df2 = df2.rename(columns={"1": "acid_1", "2": "acid_2", "3": "acid_3", "4": "acid_4", "5": "acid_5", "6": "acid_6", "7": "acid_7", "8": "acid_8", "9": "acid_9", "10": "acid_10", "11": "acid_11", "12": "acid_12", "13": "acid_13", "14": "acid_14", "15": "acid_15", "16": "acid_16", "17": "acid_17", "18": "acid_18", "19": "acid_19", "20": "acid_20", "21": "acid_21", "22": "acid_22", "23": "acid_23", "24": "acid_24", "25": "acid_25", "26": "acid_26", "27": "acid_27", "28": "acid_28", "29": "acid_29", "30": "acid_30"}, errors="raise")

  #base position
  df3 = pd.read_pickle(directoryn + '/preprocessed_basepos' + '.p')
  df3 = df3.rename(columns={"1": "base_1", "2": "base_2", "3": "base_3", "4": "base_4", "5": "base_5", "6": "base_6", "7": "base_7", "8": "base_8", "9": "base_9", "10": "base_10", "11": "base_11", "12": "base_12", "13": "base_13", "14": "base_14", "15": "base_15", "16": "base_16", "17": "base_17", "18": "base_18", "19": "base_19", "20": "base_20", "21": "base_21", "22": "base_22", "23": "base_23", "24": "base_24", "25": "base_25", "26": "base_26", "27": "base_27", "28": "base_28", "29": "base_29", "30": "base_30"}, errors="raise")

  #aroma position
  df4 = pd.read_pickle(directoryn + '/preprocessed_aromapos' + '.p')
  df4 = df4.rename(columns={"1": "aroma_1", "2": "aroma_2", "3": "aroma_3", "4": "aroma_4", "5": "aroma_5", "6": "aroma_6", "7": "aroma_7", "8": "aroma_8", "9": "aroma_9", "10": "aroma_10", "11": "aroma_11", "12": "aroma_12", "13": "aroma_13", "14": "aroma_14", "15": "aroma_15", "16": "aroma_16", "17": "aroma_17", "18": "aroma_18", "19": "aroma_19", "20": "aroma_20", "21": "aroma_21", "22": "aroma_22", "23": "aroma_23", "24": "aroma_24", "25": "aroma_25", "26": "aroma_26", "27": "aroma_27", "28": "aroma_28", "29": "aroma_29", "30": "aroma_30"}, errors="raise")

  #aroma minus acids (difference of amount)
  df5 = pd.read_pickle(directoryn + '/preprocessed_acidaromadiff' + '.p')


  df6 = pd.read_pickle(directoryn + '/preprocessed_radius' + '.p')


  df7 = pd.read_pickle(directoryn + '/preprocessed_ss3' + '.p')
  df7 = df7.rename(columns={"A": "Helix","B":"Beta","C":"Coil"})


  df8 = pd.read_pickle(directoryn + '/preprocessed_aciddist' + '.p')
  df8 = df8.rename(columns={'0 to 10':'Ac 0 to 10','10 to 20':'Ac 10 to 20','20 to 30':'Ac 20 to 30','30 to 40':'Ac 30 to 40','40 to 50':'Ac 40 to 50','50 to 60':'Ac 50 to 60','60 to 70':'Ac 60 to 70','70 to 80':'Ac 70 to 80','80 to 90':'Ac 80 to 90','90 to 100':'Ac 90 to 100','100+':'Ac 100+','Only 1':'Only 1 acid','Absent':'Acid absent'})


  df9 = pd.read_pickle(directoryn + '/preprocessed_aromadist' + '.p')
  df9 = df9.rename(columns={'0 to 10':'Ar 0 to 10','10 to 20':'Ar 10 to 20','20 to 30':'Ar 20 to 30','30 to 40':'Ar 30 to 40','40 to 50':'Ar 40 to 50','50 to 60':'Ar 50 to 60','60 to 70':'Ar 60 to 70','70 to 80':'Ar 70 to 80','80 to 90':'Ar 80 to 90','90 to 100':'Ar 90 to 100','100+':'Ar 100+','Only 1':'Only 1 aroma','Absent':'Aroma absent'})


  df10 = pd.read_pickle(directoryn + '/preprocessed_basedist' + '.p')
  df10 = df10.rename(columns={'0 to 10':'B 0 to 10','10 to 20':'B 10 to 20','20 to 30':'B 20 to 30','30 to 40':'B 30 to 40','40 to 50':'B 40 to 50','50 to 60':'B 50 to 60','60 to 70':'B 60 to 70','70 to 80':'B 70 to 80','80 to 90':'B 80 to 90','90 to 100':'B 90 to 100','100+':'B 100+','Only 1':'Only 1 base','Absent':'Base absent'})


  df11 = pd.read_pickle(directoryn + '/preprocessed_acidaromadist' + '.p')
  df11 = df11.rename(columns={'0 to 10':'AcAr 0 to 10','10 to 20':'AcAr 10 to 20','20 to 30':'AcAr 20 to 30','30 to 40':'AcAr 30 to 40','40 to 50':'AcAr 40 to 50','50 to 60':'AcAr 50 to 60','60 to 70':'AcAr 60 to 70','70 to 80':'AcAr 70 to 80','80 to 90':'AcAr 80 to 90','90 to 100':'AcAr 90 to 100','100+':'AcAr 100+','1 or both absent':'AcAr 1 or Both Absent'})


  df12 = pd.read_pickle(directoryn + '/preprocessed_acidbasedist' + '.p')
  df12 = df12.rename(columns={'0 to 10':'AcB 0 to 10','10 to 20':'AcB 10 to 20','20 to 30':'AcB 20 to 30','30 to 40':'AcB 30 to 40','40 to 50':'AcB 40 to 50','50 to 60':'AcB 50 to 60','60 to 70':'AcB 60 to 70','70 to 80':'AcB 70 to 80','80 to 90':'AcB 80 to 90','90 to 100':'AcB 90 to 100','100+':'AcB 100+','1 or both absent':'AcB 1 or Both Absent'})


  df13 = pd.read_pickle(directoryn + '/preprocessed_aromabasedist' + '.p')
  df13 = df13.rename(columns={'0 to 10':'ArB 0 to 10','10 to 20':'ArB 10 to 20','20 to 30':'ArB 20 to 30','30 to 40':'ArB 30 to 40','40 to 50':'ArB 40 to 50','50 to 60':'ArB 50 to 60','60 to 70':'ArB 60 to 70','70 to 80':'ArB 70 to 80','80 to 90':'ArB 80 to 90','90 to 100':'ArB 90 to 100','100+':'ArB 100+','1 or both absent':'ArB 1 or Both Absent'})
  
  

  
  df = df.merge(df2,on='curfile')
  df = df.merge(df3,on='curfile')
  df = df.merge(df4,on='curfile')
  df = df.merge(df5,on='curfile')
  df = df.merge(df6,on='curfile')
  df = df.merge(df7,on='curfile')
  df = df.merge(df8,on='curfile')
  df = df.merge(df9,on='curfile')
  df = df.merge(df10,on='curfile')
  df = df.merge(df11,on='curfile')
  df = df.merge(df12,on='curfile')
  df = df.merge(df13,on='curfile')

  #df_final = pd.concat([df,df2], axis=1)
  
  #print(df)

  pickle_file = open(directoryn + '/preprocessed_combined_all_fulldata.p','wb')
  pickle.dump(df,pickle_file)
  pickle_file.close()

