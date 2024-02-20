
from unicodedata import bidirectional
import torch
import torch.nn.functional as F
import torch.nn as nn
import time
from multiprocessing import Pool
from GNN.layers import GAT_gate
from tqdm import tqdm



N_atom_features = 78

class GNN_Model_atom(nn.Module):
    def __init__(self, params):
        super(GNN_Model_atom, self).__init__()
        n_graph_layer = params['n_graph_layer']
        d_graph_layer = params['d_graph_layer']
        n_FC_layer = params['n_FC_layer']
        d_FC_layer = params['d_FC_layer']
        
        self.params = params
        self.dropout_rate = params['dropout_rate']


        self.layers1 = [d_graph_layer for i in range(n_graph_layer +1)]
        self.gconv1 = nn.ModuleList \
            ([GAT_gate(self.layers1[i], self.layers1[ i +1]) for i in range(len(self.layers1 ) -1)])

        self.FC = nn.ModuleList([nn.Linear(self.layers1[-1], d_FC_layer) if i== 0 else
                                 nn.Linear(d_FC_layer, 1) if i == n_FC_layer - 1 else
                                 nn.Linear(d_FC_layer, d_FC_layer) for i in range(n_FC_layer)])
        self.mu = nn.Parameter(torch.Tensor([params['initial_mu']]).float())
        
        self.dev = nn.Parameter(torch.Tensor([params['initial_dev']]).float())
        
        if self.params['mode'] == 18:
            N_atom_features = 48
        elif self.params['mode'] == 21:
            N_atom_features = 78
        elif self.params['mode'] == 50: #one hot pos, one hot res
            N_atom_features = 50
        elif self.params['mode'] == 53: #one hot pos, one hot res, AA criteria
            N_atom_features = 53
        elif self.params['mode'] == 56:# one hot pos, one hot res, AA criteria, SS
            N_atom_features = 56
        elif self.params['mode'] == 59:#one hot pos, one hot res,AA criteria,Amphipathic-6
            N_atom_features = 59
        elif self.params['mode'] == 60:#one hot pos, one hot res,AA criteria,PLDDT
            N_atom_features = 60
        elif self.params['mode'] == 63:#one hot pos,one hot res,AA criteria,ASA
            N_atom_features = 63
        elif self.params['mode'] == 64:#one hot pos,one hot res,AA criteria,Amphi-11
            N_atom_features = 64
        elif self.params['mode'] == 77:#one hot pos,one hot res,AA criteria,Amphi-23
            N_atom_features = 77
        elif self.params['mode'] == 100:#one hot pos,one hot res,AA criteria,Amphi-26
            N_atom_features = 100
        else:
            N_atom_features = 53
            
        self.embede = nn.Linear(N_atom_features,d_graph_layer,bias=False)
        self.layernorm = nn.LayerNorm(d_graph_layer)
        


    def fully_connected(self, c_hs):
        for k in range(len(self.FC)):
            # c_hs = self.FC[k](c_hs)
            if k < len(self.FC) - 1:
                c_hs = self.FC[k](c_hs)
                c_hs = F.dropout(c_hs, p=self.dropout_rate, training=self.training)
                c_hs = F.relu(c_hs)
            else:
                c_hs = self.FC[k](c_hs)
        
        c_hs = torch.sigmoid(c_hs)

        return c_hs

    
    def Formulate_Adj2(self,c_adjs2,device=None): #self,c_adjs2,c_valid,atom_list,device=None)
        if device is not None:
            study_distance = c_adjs2.clone().detach().to(device)  # only focused on where there exist atoms, ignore the area filled with 0
            filled_value = torch.Tensor([0]).expand_as(study_distance).to(device)
        else:
            study_distance = c_adjs2.clone().detach().cuda()
            filled_value = torch.Tensor([0]).expand_as(study_distance).cuda()
        
        study_distance = torch.exp(-torch.pow(study_distance - self.mu.expand_as(study_distance), 2) / self.dev)
        
        c_adjs2 = torch.where(c_adjs2<=10, study_distance, filled_value)
        
        return c_adjs2

    def get_attention_weight(self,data):
        c_hs, c_adjs1, c_adjs2 = data
        atten1,c_hs1 = self.gconv1[0](c_hs, c_adjs1,request_attention=True)  
        atten2,c_hs2 = self.gconv1[0](c_hs, c_adjs2,request_attention=True)
        return atten1,atten2
    def embede_graph(self, data):
        c_hs, c_adjs1, c_adjs2= data
        for k in range(len(self.gconv1)):
            c_hs1 = self.gconv1[k](c_hs, c_adjs1)
            c_hs2 = self.gconv1[k](c_hs, c_adjs2)
            c_hs = c_hs2 + c_hs1
            c_hs = F.dropout(c_hs, p=self.dropout_rate, training=self.training)
        return c_hs
    def Get_Prediction(self,c_hs,atom_list):
        prediction=[]
        for batch_idx in range(len(atom_list)):
            num_atoms = int(atom_list[batch_idx])
            tmp_pred=c_hs[batch_idx,:num_atoms]
            tmp_pred=tmp_pred.sum(0)
            prediction.append(tmp_pred)
        prediction = torch.stack(prediction, 0)
        return prediction
    def train_model(self,data,device=None):
        #get data
        c_hs, c_adjs1, c_adjs2, num_atoms= data

        c_hs = self.embede(c_hs)
        
        c_adjs2=self.Formulate_Adj2(c_adjs2,device)
        c_hs=self.embede_graph((c_hs,c_adjs1,c_adjs2))
        c_hs=self.Get_Prediction(c_hs,num_atoms)
        c_hs = self.fully_connected(c_hs)
        c_hs = c_hs.view(-1)
        return c_hs
    def test_model(self, data,device=None):
        c_hs, c_adjs1, c_adjs2, num_atoms = data
        c_hs = self.embede(c_hs)
        c_adjs2 = self.Formulate_Adj2(c_adjs2, device=device)
        c_hs = self.embede_graph((c_hs, c_adjs1, c_adjs2))
        c_hs = self.Get_Prediction(c_hs, num_atoms)
        c_hs = self.fully_connected(c_hs)
        c_hs = c_hs.view(-1)
        return c_hs
    def eval_model_attention(self,data,device):
        c_hs, c_adjs1, c_adjs2, num_atoms = data
        c_hs = self.embede(c_hs)
        c_adjs2 = self.Formulate_Adj2(c_adjs2, num_atoms, device=device)
        attention1,attention2 = self.get_attention_weight((c_hs, c_adjs1, c_adjs2))
        return attention1,attention2
    def feature_extraction(self,c_hs):
        for k in range(len(self.FC)):
                # c_hs = self.FC[k](c_hs)
            if k < len(self.FC) - 1:
                c_hs = self.FC[k](c_hs)
                c_hs = F.dropout(c_hs, p=self.dropout_rate, training=False)
                c_hs = F.relu(c_hs)

            return c_hs
    def model_gnn_feature(self, data,device):
        c_hs, c_adjs1, c_adjs2, num_atoms = data
        c_hs = self.embede(c_hs)
        c_adjs2 = self.Formulate_Adj2(c_adjs2, num_atoms,device)
        # then do the gate
        c_hs = self.embede_graph((c_hs, c_adjs1, c_adjs2))
        # sum based on the atoms
        c_hs = self.Get_Prediction(c_hs, num_atoms)
        #c_hs = self.fully_connected(c_hs)
        #c_hs = c_hs.view(-1)
        c_hs=self.feature_extraction(c_hs)
        return c_hs

