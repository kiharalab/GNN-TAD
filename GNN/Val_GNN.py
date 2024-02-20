import torch
import numpy as np


def Val_GNN(model,test_dataloader,device,params):
    model.eval()

    Pred=[]
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_dataloader):
            H, A1, A2, Atom_count = sample

            batch_size = H.size(0)

            if params['mode'] == 18: # onehot residue and GNN-DOVE features
                H = torch.cat((H[...,:28], H[...,58:78]), 2)
            elif params['mode'] == 50:
                H = H[...,:50]
            elif params['mode'] == 53:
                H = H[...,:53]
            elif params['mode'] == 56:
                H = H[...,:56]
            elif params['mode'] == 59:
                H = torch.cat((H[...,:53], H[...,73:79]), 2) #onehot pos res aa + amphi-06
            elif params['mode'] == 60: #onehot pos res aa + plddt 
                H = torch.cat((H[...,:53], H[...,56:63]), 2)
            elif params['mode'] == 63: #onehot pos res aa + asa 
                H = torch.cat((H[...,:53], H[...,63:73]), 2)
            elif params['mode'] == 64:#one hot pos,one hot res,AA criteria,amphi-11
                H = torch.cat((H[...,:53], H[...,73:84]), 2)
            elif params['mode'] == 77:#one hot pos,one hot res,AA criteria,amphi-24
                H = torch.cat((H[...,:53], H[...,73:97]), 2)
            elif params['mode'] == 100:#one hot pos,one hot res,AA criteria,amphi-47
                H = torch.cat((H[...,:53], H[...,73:120]), 2)

            H, A1, A2 = H.to(device), A1.to(device), A2.to(device)
            pred = model.test_model((H, A1, A2, Atom_count),device)
            pred1=pred.detach().cpu().numpy()
            for k in range(batch_size):
                Pred.append(pred1[k])

    threshold = params['threshold']  
    pred_label=np.zeros(len(Pred))
    for k in range(len(Pred)):
        if Pred[k]>threshold:
            pred_label[k]=1
    
    return pred_label[0]




