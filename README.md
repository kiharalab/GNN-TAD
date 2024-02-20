# GNN-TAD

GNN-TAD is a Graph Neural Network based method that can be used to predict transcriptional activation domain functionality.

License: GPL v3. (If you are interested in a different license, for example, for commercial use, please contact us.)

Contact: Daisuke Kihara (dkihara@purdue.edu)

For technical questions or problems, please contact Farhanaz Farheen (ffarheen@purdue.edu)


## Pre-required Software
Python 3.9: https://www.python.org/downloads/


## Installation  
### 1. [`Install git`](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) 
### 2. Clone the repository in your computer 
```
git clone https://github.com/kiharalab/GNN-TAD && cd GNN-TAD
```

### 3. Build dependencies.   
#### 3.1 [`Install anaconda`](https://www.anaconda.com/download). 
#### 3.2 Create a new conda environment
You can create a new conda environment and activate it using the following set of commands. Make sure you are in the GNN-TAD directory and then run

```
conda create -n gnntad python=3.9
```

Each time when you want to run this code, simply activate the environment by

```
conda activate gnntad
conda deactivate    (If you want to exit) 
```


#### 3.3 [`Install EsmFold`](https://github.com/facebookresearch/esm). 
Install EsmFold from this repository. Then copy the 'esm' folder into the GNN-TAD folder. 
#### 3.4 Install dependency in command line

For installing other requirements, please use the following command:

```
pip3 install -r requirements.txt 
```


## Usage


Depending on whether you want to use atom-level GNN, residue-level GNN or logistic regression, you can specify the execution with different modes. Besides, you can also provide as input a single sequence, or a group of sequences. In all cases, the output is a file named prediction.txt containing the input sequence(s) and the corresponding binary classification value(s) (1 or 0). Here 1 means functional and 0 means non-functional.

### Mode to feature mapping

For running the code, modes need to be specified. A specific value of mode corresponds to a particular feature combination. This mode to feature mapping is given in the following table. The models are under the best_models directory.

| GNN           | Combinations  | Features                                    | Mode | Threshold | Model file name |
| ------------- | ---------------- | ------------------------------------------- | ---- | --------- | ------------ |
| Residue-level | 1                | Position of residues, type of amino acid    | 50   | 0.85      | model_50.pth.tar |
| Residue-level | 2                | Position of residues, type of amino acid, type of residue - acidic/basic/aromatic    | 53   | 0.85      | model_53.pth.tar |
| Residue-level | 3                | Position of residues, type of amino acid, type of residue - acidic/basic/aromatic, secondary structure    | 56   | 0.85      | model_56.pth.tar |
| Residue-level | 4                | Position of residues, type of amino acid, type of residue - acidic/basic/aromatic, pLDDT    | 60   | 0.85      | model_60.pth.tar |
| Residue-level | 5        | Position of residues, type of amino acid, type of residue - acidic/basic/aromatic, relative accessible surface area    | 63   | 0.85 | model_63.pth.tar |
| Residue-level | 6 (bin-size = 1) | Position of residues, type of amino acid, type of residue - acidic/basic/aromatic, amphipathic index (bin-size = 1)   | 59   | 0.8 | model_59.pth.tar |
| Residue-level | 6 (bin-size = 0.5) | Position of residues, type of amino acid, type of residue - acidic/basic/aromatic, amphipathic index (bin-size = 0.5)   | 64   | 0.85      | model_64.pth.tar |
| Residue-level | 6 (bin-size = 0.2) | Position of residues, type of amino acid, type of residue - acidic/basic/aromatic, amphipathic index (bin-size = 0.2)   | 77   | 0.85      | model_77.pth.tar |
|Residue-level|6 (bin-size = 0.1) | Position of residues, type of amino acid, type of residue - acidic/basic/aromatic, amphipathic index (bin-size = 0.1)| 100| 0.8| model_100.pth.tar |
| Atom-level    | 1                  | GNN-DOVE features, type of amino acid                                                            | 18   | 0.85      | model_18.pth.tar |
| Atom-level    | 2                  | GNN-DOVE features, type of amino acid, position of residues                                          | 21   | 0.8       | model_21.pth.tar |
| Logistic Regression    | 1         | Sequence-only                                                                                            | 1   | 0.85      | logreg_mode1.npz |
| Logistic Regression    | 2         | Structure-only                                                                                             | 2   | 0.85      | logreg_mode2.npz |
| Logistic Regression    | 3         | Sequence & structure combined                                                                               | 3   | 0.85     | logreg_mode3.npz |

### Command to run the code
```
usage: main.py [-h] [--esmfold_model_path ESMFOLD_MODEL_PATH]
               [--esmfold_output ESMFOLD_OUTPUT] [--dssp_output DSSP_OUTPUT]
               [--input_csv INPUT_CSV] [--multi_mode MULTI_MODE]
               [--single_seq SINGLE_SEQ] [--ss3_csv SS3_CSV] [--mode MODE]
               [--type TYPE] [--gpu GPU] [--num_workers NUM_WORKERS]
               [--n_graph_layer N_GRAPH_LAYER] [--d_graph_layer D_GRAPH_LAYER]
               [--n_FC_layer N_FC_LAYER] [--d_FC_layer D_FC_LAYER]
               [--modelpath MODELPATH] [--initial_mu INITIAL_MU]
               [--initial_dev INITIAL_DEV] [--seed SEED] --threshold THRESHOLD

optional arguments:
  -h, --help            show this help message and exit
  --esmfold_model_path  path to EsmFold model
  --esmfold_output      esmfold output path
  --dssp_output         dssp output path
  --input_csv           input path - needed for multiple input mode
  --multi_mode          mode for selecting single or multiple sequence input;
                        0 for single, 1 for multi
  --single_seq          sequence input for single sequence mode
  --ss3_csv             ss3 csv path
  --mode                Choose feature combination type
  --type                Choose residue or atom level. 1 - residue, 2 - atom, 3 - logistic regression
  --gpu                 Choose gpu id, example: '1,2'(specify use gpu 1 and 2)
  --num_workers         number of workers
  --n_graph_layer       number of GNN layer
  --d_graph_layer       dimension of GNN layer
  --n_FC_layer          number of FC layer
  --d_FC_layer          dimension of FC layer
  --modelpath           GNN or logistic regression model path
  --initial_mu          initial value of mu
  --initial_dev         initial value of dev
  --seed                random seed for shuffling
  --threshold           threshold for predicting the binary value

```

### Examples
Example commands to run the code with different modes or types are shown below. 

#### 1. Residue-level GNN
For residue-level GNN, the type should be set to 1. The modes should be among the residue-GNN modes that can be found in the mode-to-feature mapping table above.

##### 1.1 Single input
For single input, you need to specify the sequence in the command line while setting multi_mode to 0. 

###### Example 
```
python3 main.py --single_seq=IGIRTIVADVGISVPFVTIDVGVEEFYCMI --multi_mode=0 --mode=53 --type=1 --modelpath=best_models/model_53.pth.tar --threshold=0.85 --gpu=1
```

##### 1.2 Multiple inputs
For multiple inputs, you need to provide a csv file as input with two columns. The first one is the id, and the second one is the sequence (aa_seq). A sample scenario has been provided in data_sample/dataset_sample.csv. For multiple inputs, just type the sequences under the aa_seq column and the id values are for specifying the sequence id which can simply be sequential numbers like 1,2,3... Note that, two distinct sequences should not be provided with the same id. The multi_mode should be set to 1.

###### Example 
```
python3 main.py --input_csv=data_sample/dataset_sample.csv --multi_mode=1 --mode=53 --type=1 --modelpath=best_models/model_53.pth.tar --threshold=0.85 --gpu=1
```


#### 2. Atom-level GNN
For atom-level GNN, the type should be set to 2. The modes should be among the atom-GNN modes that can be found in the mode-to-feature mapping table above.

##### 2.1 Single input
This is the same as 1.1 with type set to 2 and mode set to 18 or 21. The model file should also be corresponding to the atom-level GNN mode.

###### Example 
```
python3 main.py --single_seq=IGIRTIVADVGISVPFVTIDVGVEEFYCMI --multi_mode=0 --mode=18 --type=2 --modelpath=best_models/model_18.pth.tar --threshold=0.85 --gpu=1
```

##### 2.2 Multiple inputs
This is the same as 1.2 but with type set to 2 and mode set to either 18 or 21. The model file should also be corresponding to the atom-level GNN mode.

###### Example 
```
python3 main.py --input_csv=data_sample/dataset_sample.csv --multi_mode=1 --mode=18 --type=2 --modelpath=best_models/model_18.pth.tar --threshold=0.85 --gpu=1
```

#### 3. Logistic Regression
For logistic regression, the type should be set to 3. The modes should be among the logistic regression modes that can be found in the mode-to-feature mapping table above.

##### 3.1 Single input
This is the same as 1.1 with type set to 3 and mode set to 1, 2 or 3. The model file should also be corresponding to the logistic regression mode.

###### Example 
```
python3 main.py --single_seq=IGIRTIVADVGISVPFVTIDVGVEEFYCMI --multi_mode=0 --mode=1 --type=3 --modelpath=best_models/logreg_mode1.npz --threshold=0.85 --gpu=1
```

##### 3.2 Multiple inputs
This is the same as 1.2 but with type set to 3 and mode set to 1, 2 or 3. The model file should also be corresponding to the logistic regression mode.

###### Example 
```
python3 main.py --input_csv=data_sample/dataset_sample.csv --multi_mode=1 --mode=1 --type=3 --modelpath=best_models/logreg_mode1.npz --threshold=0.85 --gpu=1
```

### Input

As mentioned above, input can be provided in two ways: with single sequences and with multiple sequences given in a csv file, a sample of which has been given in the data_sample directory. The input should be 30-length amino acid sequences.

### Output

In all cases, the output is a text file named predictions.txt that will contain the sequence and the corresponding prediction value (1 or 0) where 1 means functional and 0 means non-functional. Besides, the DSSP outputs will be under the 'dssp_outputs' folder while the EsmFold predictions will be under the 'predicted_structures' folder.
