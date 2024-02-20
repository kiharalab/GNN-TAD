#
# Copyright (C) 2020 Xiao Wang
# Email:xiaowang20140001@gmail.com wang3702@purdue.edu
#

from email.policy import default

import argparse

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--esmfold_model_path',type=str,default="esmfold_model")
    parser.add_argument('--esmfold_output',type=str,default="predicted_structures",help='decoy esmfold output path')
    parser.add_argument('--dssp_output',type=str,default="dssp_outputs",help='decoy dssp output path')
    parser.add_argument('--input_csv',type=str, default="input_csv.csv",help='decoy input path')
    parser.add_argument('--multi_mode',type=int, default=1,help='mode for selecting single or multiple sequence input; 0 for single, 1 for multi')
    parser.add_argument('--single_seq',type=str, default="GGGGGGGGGGGGGGGGGGGGGGGGGGGGGG",help='sequence input for single sequence mode')
    parser.add_argument('--ss3_csv',type=str, default="aaseq_ss3",help='decoy ss3 csv path')
    parser.add_argument('--mode',type=int,default=53,help='Choose feature combination type') # to control which features are used for training
    parser.add_argument('--type',type=int,default=1,help='Choose residue or atom level. 1 - residue, 2 - atom, 3 - logistic regression') # to control atom-level or residue-level
    parser.add_argument('--gpu',type=str,default='1',help='Choose gpu id, example: \'1,2\'(specify use gpu 1 and 2)')
    parser.add_argument("--num_workers", help="number of workers", type=int, default=4)
    parser.add_argument("--n_graph_layer", help="number of GNN layer", type=int, default=4)
    parser.add_argument("--d_graph_layer", help="dimension of GNN layer", type=int, default=140)
    parser.add_argument("--n_FC_layer", help="number of FC layer", type=int, default=4)
    parser.add_argument("--d_FC_layer", help="dimension of FC layer", type=int, default=128)
    parser.add_argument("--modelpath", help="model path", type=str, default='trained_model/model_53.pth.tar')
    parser.add_argument("--initial_mu", help="initial value of mu", type=float, default=0.0)
    parser.add_argument("--initial_dev", help="initial value of dev", type=float, default=0.0)
    parser.add_argument("--dropout_rate", help="dropout_rate", type=float, default=0.3)
    parser.add_argument('--portion1',help='percent of data used for training while others for validation',type=float,default=0.75)
    parser.add_argument('--portion2',help='percent of data upto which validation set is chosen',type=float,default=0.9)
    parser.add_argument('--seed',type=int,default=888,help='random seed for shuffling')
    parser.add_argument("--reg",type=float,default=1e-5,help="training regularization term")
    parser.add_argument("--weight_decay",type=float,default=1e-4,help="weight decay for model")
    parser.add_argument("--clip",type=float,default=1,help="gradient clip for training")
    parser.add_argument("--lstm",type=int,default=0,help="use lstm to do position encoding")
    parser.add_argument("--positional", type=int, default=0, help="if enable postional encoder")
    parser.add_argument("--onehot", type=int, default=0, help="if enable onehot for positional and residue type")
    parser.add_argument("--draw_roc", type=int, default=0, help="1: while draw roc mode is on")
    #parser.add_argument('--save_path',type=str, required=True,help='log path')
    parser.add_argument('--threshold',type=float, required=True,help='threshold for predicting the binary value')
    #Dense points part parameters
    args = parser.parse_args()
    # try:
    #     import ray,socket
    #     rayinit()
    # except:
    #     print('ray need to be installed')#We do not need this since GAN can't be paralleled.
    params = vars(args)
    return params
    