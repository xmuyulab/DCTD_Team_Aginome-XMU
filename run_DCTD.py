# Source Code for submissions to DREAM Challenge Tumor Deconvolution
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from sklearn import preprocessing as pp
import pandas as pd
import argparse
import os


parser = argparse.ArgumentParser(description='DCTD_Team Aginome-XMU.')
subparsers = parser.add_subparsers(dest='subcommand', help='Select one of the following sub-commands')

parser_a = subparsers.add_parser('coarse', help='coarse-grained deconvolution',description="coarse-grained deconvolution")
parser_a.add_argument("-In", type=str, help="Input expression file (with genes specified using HUGO symbols)", default="./expr.csv")
parser_a.add_argument("-Out", type=str, help="Output result file", default="./predicton.csv")
parser_a.add_argument("-scale", type=str, help="The scale of the expression data (i.e., Log2, Log10, Linear)", default="Linear")
parser_a.add_argument("-model", type=str, help="Trained models directory", default="./model/")
parser_a.add_argument("-dataset", type=str, help="name of test dataset", default="test")



parser_b = subparsers.add_parser('fine', help='fine-grained deconvolution',description="fine-grained deconvolution")
parser_b.add_argument("-In", type=str, help="Input expression file (with genes specified using HUGO symbols)", default="expr.csv")
parser_b.add_argument("-Out", type=str, help="Output result file", default="./predicton.csv")
parser_b.add_argument("-scale", type=str, help="The scale of the expression data (i.e., Log2, Log10, Linear)", default="Linear")
parser_b.add_argument("-model", type=str, help="Trained models directory", default="./model/")
parser_b.add_argument("-dataset", type=str, help="Name of test dataset", default="test")



def sample_scaling(x, scaling_option):
    """
    Apply scaling of data
    :param x:
    :param scaling_option:
    :return:
    """

    if scaling_option == "log_min_max":
        # Bring in log space
        x = np.log2(x + 1)

        # Normalize data
        mms = pp.MinMaxScaler(feature_range=(0, 1), copy=True)

        # it scales features so transpose is needed
        x = mms.fit_transform(x.T).T

    return x
        
def MCpred(model,data):
    data = sample_scaling(data.T,"log_min_max").T
    data = data.T
    data = torch.from_numpy(data)
    model.eval()
    pred = model(data)
    return pred

class MLP_coarse(torch.nn.Module):  
    def __init__(self,INPUT_SIZE,OUTPUT_SIZE):
        super(MLP_coarse, self).__init__()  
        L1 = 256
        L2 = 512
        L3 = 128
        L4 = 32
        L5 = 16
        self.hidden = torch.nn.Sequential(                       
            nn.Linear(INPUT_SIZE, L1),
            nn.Tanh(),
            nn.Linear(L1,L2),
            nn.BatchNorm1d(L2),
            nn.ReLU(),
            nn.Linear(L2,L3),
            nn.Tanh(),
            nn.Linear(L3,L4),
            nn.ReLU(),
            nn.Linear(L4,L5),
            nn.Tanh(),
        )
        self.predict =  torch.nn.Sequential( 
            nn.Linear(L5, OUTPUT_SIZE),
        )
    def forward(self, x):   
        y = self.hidden(x)    
        y = self.predict(y)   
        return y

class MLP_fine1(torch.nn.Module):  
    def __init__(self,INPUT_SIZE,OUTPUT_SIZE):
        super(MLP_fine1, self).__init__()     
        
        L1 = 1024
        L2 = 512
        L3 = 256
        L4 = 128
        L5 = 32
        self.hidden = torch.nn.Sequential(                       
            nn.Linear(INPUT_SIZE, L1),
            nn.Tanh(),
            nn.Linear(L1,L2),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(L2,L3),
            nn.Tanh(),
            nn.Linear(L3,L4),
            nn.ReLU(),
            nn.Linear(L4,L5),
            nn.Tanh(),
        )
        self.predict =  torch.nn.Sequential( 
            nn.Linear(L5, OUTPUT_SIZE),
        )
    def forward(self, x):   
        
        y = self.hidden(x)     
        y = self.predict(y)    
        return y

class MLP_fine2(torch.nn.Module):  
    def __init__(self,INPUT_SIZE,OUTPUT_SIZE):
        super(MLP_fine2, self).__init__()     
        L1 = 1024
        L2 = 1024
        L3 = 512
        L4 = 256
        L5 = 32
        self.hidden = torch.nn.Sequential(                       
            nn.Linear(INPUT_SIZE, L1),
            nn.Tanh(),
            nn.Linear(L1,L2),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(L2,L3),
            nn.Tanh(),
            nn.Linear(L3,L4),
            nn.ReLU(),
            nn.Linear(L4,L5),
            nn.Tanh(),
        )
        self.predict =  torch.nn.Sequential( 
            nn.Linear(L5, OUTPUT_SIZE),
        )
    def forward(self, x):   
        
        y = self.hidden(x)    
        y = self.predict(y)    
        return y


def do_MLP_coarse(expression_paths,scale,modelpath,dataset_name):
    exp = pd.read_csv(expression_paths, sep=",", index_col=0)
    print('Load data!')
    if scale == "Log2":
        exp = pow(2,exp)

    if scale == "Log10":
        exp = pow(10,exp)
        
    if any(exp<0):
        exp = exp + abs(exp.min().min())
    
    exp_data = exp.reindex(sig_genes).values

    final_result = np.zeros(shape=(8,exp_data.shape[1]))
    print('Prediction started!')
    for c, celltype in enumerate(celltypes):
        file_list = os.listdir(modelpath+celltype)
        for i,file in enumerate(file_list):
            if c == 0 or c ==4 or c ==5:
                model1.load_state_dict(torch.load(modelpath+'/'+celltype+'/'+file,map_location='cpu'))
                model = model1
                cc = 8
            else:
                model2.load_state_dict(torch.load(modelpath+'/'+celltype+'/'+file,map_location='cpu'))
                model = model2 
                cc =9
                
            out = MCpred(model=model,data=exp_data)
            
            pred = Variable(out,requires_grad=False).cpu().numpy().reshape(exp_data.shape[1],cc)
            if i == 0:
                final_pred = pred.T[c]
            else:
                final_pred += pred.T[c]
        
        final_result[c] = final_pred.T/len(file_list)

    pred_result = pd.DataFrame(final_result,index=celltypes,columns=exp.columns)
    
    pred_result['cell.type']=pred_result.index
    result = pd.melt(pred_result,id_vars='cell.type',var_name='sample.id',value_name='prediction')
    result['dataset.name']=dataset_name
    
    return result

def do_MLP_fine(expression_paths,scale,modelpath,dataset_name):
    exp = pd.read_csv(expression_paths, sep=",", index_col=0)
    print('Load data!')
    if scale == "Log2":
        exp = pow(2,exp)

    if scale == "Log10":
        exp = pow(10,exp)
        
    if any(exp<0):
        exp = exp + abs(exp.min().min())
    
    exp_data = exp.reindex(sig_genes).values

    final_result = np.zeros(shape=(14,exp_data.shape[1]))

    print('Prediction started!')
    for c, celltype in enumerate(celltypes):
        file_list = os.listdir(modelpath+celltype)
        for i,file in enumerate(file_list):
            if c == 4 or c ==5 or c ==6 or c ==7:
                model1.load_state_dict(torch.load(modelpath+"/"+celltype+'/'+file,map_location='cpu'))
                model = model1 
            else:
                model2.load_state_dict(torch.load(modelpath+"/"+celltype+'/'+file,map_location='cpu'))
                model = model2 
                
            out = MCpred(model=model,data=exp_data)
            
            pred = Variable(out,requires_grad=False).cpu().numpy().reshape(exp_data.shape[1],len(celltypes))
            if i == 0:
                final_pred = pred.T[c]
            else:
                final_pred += pred.T[c]
        
        final_result[c] = final_pred.T/len(file_list)

    pred_result = pd.DataFrame(final_result,index=celltypes,columns=exp.columns)
    
    pred_result['cell.type']=pred_result.index
    result = pd.melt(pred_result,id_vars='cell.type',var_name='sample.id',value_name='prediction')
    result['dataset.name']=dataset_name
    
    return result

    

if __name__ == '__main__':

    inputArgs = parser.parse_args()
    
    sig_genes = pd.read_csv('validation_features_5080.txt',sep="\t",index_col=0)
    sig_genes= list(sig_genes['feature_left'])
    sig_genes.sort()

    if (inputArgs.subcommand=='coarse'):
        celltypes = ['B.cells', 'CD4.T.cells', 'CD8.T.cells', 'NK.cells','monocytic.lineage','neutrophils', 'fibroblasts','endothelial.cells']

        torch.manual_seed(0)
        model1 = MLP_coarse(INPUT_SIZE=len(sig_genes),OUTPUT_SIZE=len(celltypes)).double()
        model2 = MLP_coarse(INPUT_SIZE=len(sig_genes),OUTPUT_SIZE=9).double()

        result = do_MLP_coarse(inputArgs.In, inputArgs.scale,inputArgs.model,inputArgs.dataset)
    
    if (inputArgs.subcommand=='fine'):
        celltypes = ['naive.B.cells', 'memory.B.cells', 'naive.CD4.T.cells','memory.CD4.T.cells','regulatory.T.cells', 'naive.CD8.T.cells','memory.CD8.T.cells', 'NK.cells', 'monocytes','myeloid.dendritic.cells', 'macrophages', 'neutrophils', 'fibroblasts','endothelial.cells']

        torch.manual_seed(0)
        model1 = MLP_fine1(INPUT_SIZE=len(sig_genes),OUTPUT_SIZE=len(celltypes)).double()
        model2 = MLP_fine2(INPUT_SIZE=len(sig_genes),OUTPUT_SIZE=len(celltypes)).double()

        result = do_MLP_fine(inputArgs.In, inputArgs.scale,inputArgs.model,inputArgs.dataset)

    result.to_csv(inputArgs.Out, sep=',', index=False)

    print('Prediction finished!')
    