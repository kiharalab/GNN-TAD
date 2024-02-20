import pandas as pd
import os
from tqdm import tqdm

import numpy as np
import sys
import shutil


def add_cryst1(filename,input_path,temp_dir):
    from_file = open(input_path,'r')
    from_file.readline() # and discard
    try:
        os.mkdir(temp_dir)
    except:
        pass
    to_file = open(temp_dir + '/' + filename,'w')
    to_file.write('CRYST1\n')
    shutil.copyfileobj(from_file, to_file)
    from_file.close()
    to_file.close()

def run_dssp_prog(pdb_file,input_path,output_path):
    os.system('dssp ' + input_path + ' ' + output_path + '/' + pdb_file[:-4] + '.dssp')

def prep_input_for_dssp(input_path,temp_path):
    listfiles = os.listdir(input_path)
    for eachfile in tqdm(listfiles):
        add_cryst1(eachfile,input_path + "/" + eachfile,temp_path)
        

def do_dssp(input_path,output_path):
    try:
        os.mkdir(output_path)
    except:
        pass
    listfiles = os.listdir(input_path)
    for eachfile in tqdm(listfiles):
        run_dssp_prog(eachfile,input_path + "/" + eachfile,output_path)

def dssp_run(pdb_input,temp_dir,dssp_output):   
    prep_input_for_dssp(pdb_input,temp_dir)
    do_dssp(temp_dir,dssp_output)  
    










