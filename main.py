from ops.argparser import argparser
from GNN.Generate_GNN_Model import Gen_GNN_Model
from Preprocessing.esm_predict import predict_structure
from Preprocessing.run_dssp import dssp_run
from Preprocessing.parse_dssp import parse_dssp_files
from Preprocessing.prepare_input import preprocess_begin
from Preprocessing.onehot_SS import combine_features_SS
from Preprocessing.onehot_plddt import feature_dump_plddt
from Preprocessing.onehot_ASA import feature_dump_asa
from Preprocessing.combine_feats import combine_features
from Preprocessing.onehot_amphi import feature_dump_amphi
from Preprocessing.combine_feats_amphi import combine_features_amphi
from Preprocessing.prepare_input_atom import call_prepare

#logistic regression preprocessing
from Preprocessing.logreg.preprocess_frac_final import feature_dump_exactcount
from Preprocessing.logreg.preprocess_acidpos import feature_dump_acidpos
from Preprocessing.logreg.preprocess_basepos import feature_dump_basepos
from Preprocessing.logreg.preprocess_aromapos import feature_dump_aromapos
from Preprocessing.logreg.preprocess_acidaromadiff import feature_dump_acidaromadiff
from Preprocessing.logreg.preprocess_radius import feature_dump_radius
from Preprocessing.logreg.preprocess_ss3 import feature_dump_ss3
from Preprocessing.logreg.preprocess_aciddist import feature_dump_aciddist
from Preprocessing.logreg.preprocess_basedist import feature_dump_basedist
from Preprocessing.logreg.preprocess_aromadist import feature_dump_aromadist
from Preprocessing.logreg.preprocess_acidaromadist import feature_dump_acidaromadist
from Preprocessing.logreg.preprocess_acidbasedist import feature_dump_acidbasedist
from Preprocessing.logreg.preprocess_aromabasedist import feature_dump_aromabasedist
from Preprocessing.logreg.combined_all_fulldata import preprocess_logreg
from LogReg.test_logreg import logreg_evaluate
import os
import shutil
import pandas as pd


def run_predictions_bulk(prepped_files,GNNmode):
    params = argparser()
    dir_path = prepped_files
    model_path = os.path.abspath(params['modelpath'])
    if GNNmode!=3:
        os.environ['CUDA_VISIBLE_DEVICES'] = params['gpu']
        Gen_GNN_Model(dir_path,model_path,params)
    else:
        logreg_evaluate(params['mode'],dir_path,params['threshold'],model_path,params['ss3_csv']+'.csv')

def run_esmfold_bulk(esmfold_output):    
    params = argparser()
    input_csv = params['input_csv']
    esmfold_model_path = params['esmfold_model_path']
    predict_structure(input_csv,esmfold_output,esmfold_model_path)
    

def preprocess_all_logreg(csvpath,output_path,filepath):
    feature_dump_exactcount(csvpath,output_path)
    feature_dump_acidpos(csvpath,output_path)
    feature_dump_basepos(csvpath,output_path)
    feature_dump_aromapos(csvpath,output_path)
    feature_dump_acidaromadiff(csvpath,output_path)
    feature_dump_radius(csvpath,output_path,filepath)
    feature_dump_ss3(csvpath,output_path)
    feature_dump_aciddist(csvpath,output_path,filepath)
    feature_dump_basedist(csvpath,output_path,filepath)
    feature_dump_aromadist(csvpath,output_path,filepath)
    feature_dump_acidaromadist(csvpath,output_path,filepath)
    feature_dump_acidbasedist(csvpath,output_path,filepath)
    feature_dump_aromabasedist(csvpath,output_path,filepath)
    preprocess_logreg(output_path)


def preprocess_bulk(GNNmode):
    prepped_files = ""
    params = argparser()
    os.environ['CUDA_VISIBLE_DEVICES'] = params['gpu']
    mode = params['mode']
    esmfold_output = params['esmfold_output']
    dssp_output = params['dssp_output']
    ss3_csv = params['ss3_csv']
    run_esmfold_bulk(esmfold_output)
    dssp_run(esmfold_output,"dssp_temp_files",dssp_output)
    parse_dssp_files(dssp_output,ss3_csv)
    shutil.rmtree("dssp_temp_files")
    if GNNmode == 1:
        #residue-level GNN
        preprocess_begin(esmfold_output,"prep_pos_res_aa")
        combine_features_SS(ss3_csv+'.csv',"prep_pos_res_aa","prep_pos_res_aa_ss")
        amphi_modes = [59,64,77,100]
        feature_dump_plddt(esmfold_output,"plddt_only")
        feature_dump_asa(dssp_output,"asa_only")
        combine_features("plddt_only","asa_only","prep_pos_res_aa_ss","prep_pos_res_aa_ss_plddt_asa")
        shutil.rmtree("plddt_only")
        shutil.rmtree("asa_only")
        shutil.rmtree("prep_pos_res_aa")
        shutil.rmtree("prep_pos_res_aa_ss")
        
        prepped_files = "prep_pos_res_aa_ss_plddt_asa"
        if mode in amphi_modes: 
            feature_dump_amphi(ss3_csv+'.csv',"only_amphi_"+str(mode),mode)
            combine_features_amphi("only_amphi_"+str(mode),"prep_pos_res_aa_ss_plddt_asa","prep_pos_res_aa_ss_plddt_amphi_"+str(mode))
            shutil.rmtree("prep_pos_res_aa_ss_plddt_asa")
            shutil.rmtree("only_amphi_"+str(mode))
            prepped_files = "prep_pos_res_aa_ss_plddt_amphi_"+str(mode)
    elif GNNmode == 2:
        #atom-level GNN
        call_prepare(esmfold_output,"prep_dove_pos_res")
        prepped_files = "prep_dove_pos_res"
    elif GNNmode == 3: 
        #logistic regression
        preprocess_all_logreg(ss3_csv+'.csv',"prep_logreg_data",esmfold_output)
        prepped_files = "prep_logreg_data"
    return prepped_files



def delete_pre_files(params):
    try:
        shutil.rmtree(params['dssp_output'])
        shutil.rmtree(params['esmfold_output'])
        os.remove(params['ss3_csv']+'.csv')
        os.remove('predictions.txt')
    except:
        pass
    try:
        if int(params['multi_mode']) == 0:
            os.remove(params['input_csv'])
    except:
        pass
    try:
        shutil.rmtree('prep_pos_res_aa_ss_plddt_asa')
    except:
        pass

    try:
        shutil.rmtree('prep_pos_res_aa_ss_plddt_amphi_59')
    except:
        pass

    try:
        shutil.rmtree('prep_pos_res_aa_ss_plddt_amphi_64')
    except:
        pass

    try:
        shutil.rmtree('prep_pos_res_aa_ss_plddt_amphi_77')
    except:
        pass

    try:
        shutil.rmtree('prep_pos_res_aa_ss_plddt_amphi_100')
    except:
        pass

    try:
        shutil.rmtree('prep_dove_pos_res')
    except:
        pass

    try:
        shutil.rmtree('prep_logreg_data')
    except:
        pass


def main():
    params = argparser()
    GNNmode = params['type']
    multi_mode = params['multi_mode']
    delete_pre_files(params)
    if int(multi_mode) == 1:
        prepped_files = preprocess_bulk(GNNmode)
        run_predictions_bulk(prepped_files,GNNmode)
        print("All done bulk")
    else:
        input_csv = params['input_csv']
        sequence_input = params['single_seq']
        header = ['id', 'aa_seq']
        data_seq = [['1',sequence_input]]
        data_csv = pd.DataFrame(data_seq, columns=header)
        data_csv.to_csv(input_csv, index=True)
        prepped_files = preprocess_bulk(GNNmode)
        run_predictions_bulk(prepped_files,GNNmode)
        print("All done single")
    
if __name__ == "__main__":
    main()
    


















