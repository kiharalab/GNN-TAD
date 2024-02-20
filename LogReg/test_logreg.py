import pandas as pd
import numpy as np
import sys
import os
import pickle
from sklearn import metrics

import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")





def transform_df(df):
  #print(df.columns.values.tolist())
  df = df.drop(columns=['Ac 90 to 100','B 90 to 100','Ar 90 to 100','AcB 90 to 100','AcAr 90 to 100','ArB 90 to 100','Ac 100+','Ar 100+','B 100+','AcAr 100+','AcB 100+','ArB 100+'])
  df['Ac less than 2'] = df[['Only 1 acid', 'Acid absent']].sum(axis=1)
  df['Ar less than 2'] = df[['Only 1 aroma', 'Aroma absent']].sum(axis=1)
  df['B less than 2'] = df[['Only 1 base', 'Base absent']].sum(axis=1)
  df['Ac less than 2'] = (df['Ac less than 2'] > 0).astype(int)
  df['B less than 2'] = (df['B less than 2'] > 0).astype(int)
  df['Ar less than 2'] = (df['Ar less than 2'] > 0).astype(int)
  df = df.drop(columns = ['Only 1 acid','Acid absent','Only 1 base','Base absent','Only 1 aroma','Aroma absent'])
  return df

def logreg_evaluate(mode,prep_path,threshold,model_path,ss3_csv):

  colvals = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'acid_1', 'acid_2', 'acid_3', 'acid_4', 'acid_5', 'acid_6', 'acid_7', 'acid_8', 'acid_9', 'acid_10', 'acid_11', 'acid_12', 'acid_13', 'acid_14', 'acid_15', 'acid_16', 'acid_17', 'acid_18', 'acid_19', 'acid_20', 'acid_21', 'acid_22', 'acid_23', 'acid_24', 'acid_25', 'acid_26', 'acid_27', 'acid_28', 'acid_29', 'acid_30', 'base_1', 'base_2', 'base_3', 'base_4', 'base_5', 'base_6', 'base_7', 'base_8', 'base_9', 'base_10', 'base_11', 'base_12', 'base_13', 'base_14', 'base_15', 'base_16', 'base_17', 'base_18', 'base_19', 'base_20', 'base_21', 'base_22', 'base_23', 'base_24', 'base_25', 'base_26', 'base_27', 'base_28', 'base_29', 'base_30', 'aroma_1', 'aroma_2', 'aroma_3', 'aroma_4', 'aroma_5', 'aroma_6', 'aroma_7', 'aroma_8', 'aroma_9', 'aroma_10', 'aroma_11', 'aroma_12', 'aroma_13', 'aroma_14', 'aroma_15', 'aroma_16', 'aroma_17', 'aroma_18', 'aroma_19', 'aroma_20', 'aroma_21', 'aroma_22', 'aroma_23', 'aroma_24', 'aroma_25', 'aroma_26', 'aroma_27', 'aroma_28', 'aroma_29', 'aroma_30', '<-10', '-10 to -5', '-5 to 0', '0', '0 to +5', '+5 to +10', '>+10', 'R5-10', 'R10-15', 'R15-20', 'R20-25', 'R25-30', 'Helix', 'Beta', 'Coil', 'Ac 0 to 10', 'Ac 10 to 20', 'Ac 20 to 30', 'Ac 30 to 40', 'Ac 40 to 50', 'Ac 50 to 60', 'Ac 60 to 70', 'Ac 70 to 80', 'Ac 80 to 90', 'Ar 0 to 10', 'Ar 10 to 20', 'Ar 20 to 30', 'Ar 30 to 40', 'Ar 40 to 50', 'Ar 50 to 60', 'Ar 60 to 70', 'Ar 70 to 80', 'Ar 80 to 90', 'B 0 to 10', 'B 10 to 20', 'B 20 to 30', 'B 30 to 40', 'B 40 to 50', 'B 50 to 60', 'B 60 to 70', 'B 70 to 80', 'B 80 to 90', 'AcAr 0 to 10', 'AcAr 10 to 20', 'AcAr 20 to 30', 'AcAr 30 to 40', 'AcAr 40 to 50', 'AcAr 50 to 60', 'AcAr 60 to 70', 'AcAr 70 to 80', 'AcAr 80 to 90', 'AcAr 1 or Both Absent', 'AcB 0 to 10', 'AcB 10 to 20', 'AcB 20 to 30', 'AcB 30 to 40', 'AcB 40 to 50', 'AcB 50 to 60', 'AcB 60 to 70', 'AcB 70 to 80', 'AcB 80 to 90', 'AcB 1 or Both Absent', 'ArB 0 to 10', 'ArB 10 to 20', 'ArB 20 to 30', 'ArB 30 to 40', 'ArB 40 to 50', 'ArB 50 to 60', 'ArB 60 to 70', 'ArB 70 to 80', 'ArB 80 to 90', 'ArB 1 or Both Absent', 'Ac less than 2', 'Ar less than 2', 'B less than 2']

  #directoryn = '/home/kihara/ffarheen/GNN_whole_data_esm/logReg_final/preprocess/sampled_sets/final_data_splits_2024'
  
  file_seq_map = {}
  df_map = pd.read_csv(ss3_csv)
  for index, row in df_map.iterrows():
      file_seq_map[row['file'][:-4]] = row['aa_seq']

  prediction_str = "Sequence, Prediction"
  df_test = pd.read_pickle(prep_path+'/preprocessed_combined_all_fulldata.p')

  X_test = transform_df(df_test)
  #X_test = df_test.drop(columns=['curfile'])


  exact_count = colvals[0:20]
  acidpos = colvals[20:50]
  basepos = colvals[50:80]
  aromapos = colvals[80:110]
  acidaromadiff = colvals[110:117]
  radius = colvals[117:122]
  ss3 = colvals[122:125]
  aciddist = colvals[125:134]
  aciddist = np.append(aciddist,colvals[-3])
  aromadist = colvals[134:143]
  aromadist = np.append(aromadist,colvals[-2])
  basedist = colvals[143:152]
  basedist = np.append(basedist,colvals[-1])
  acidaromadist = colvals[152:162]
  acidbasedist = colvals[162:172]
  aromabasedist = colvals[172:182]
  
  if mode == 1:#only sequence
    X_test = X_test.drop(radius,axis=1)
    X_test = X_test.drop(ss3,axis=1)
    X_test = X_test.drop(aciddist,axis=1)
    X_test = X_test.drop(basedist,axis=1)
    X_test = X_test.drop(aromadist,axis=1)
    X_test = X_test.drop(acidaromadist,axis=1)
    X_test = X_test.drop(acidbasedist,axis=1)
    X_test = X_test.drop(aromabasedist,axis=1)
  elif mode == 2:#only structure
    X_test = X_test.drop(exact_count,axis=1)
    X_test = X_test.drop(acidpos,axis=1)
    X_test = X_test.drop(basepos,axis=1)
    X_test = X_test.drop(aromapos,axis=1)
    X_test = X_test.drop(acidaromadiff,axis=1)
  

  
    
  

  np.random.seed(42)


  # Initialize model
  current_coef = np.load(model_path)['C']
  current_intercept = np.load(model_path)['I']

  #loop here for all test samples

  for i in range(len(X_test)):
    df_each_test = X_test.iloc[i]
    curfile = df_each_test[0]
    x_test = df_each_test[1:]
    logits_2 = np.dot(x_test, current_coef) + current_intercept

    # Apply the sigmoid function to get probabilities
    probabilities_2 = 1 / (1 + np.exp(-logits_2))

    predictions_2 = (probabilities_2 >= threshold).astype(int)

    

    #print("appending to prediction file")
    prediction_str += "\n"+file_seq_map[str(int(curfile))] + "," + str(predictions_2[0])

    f1 = open('predictions.txt','w')
    f1.write(prediction_str) 
    f1.close()



