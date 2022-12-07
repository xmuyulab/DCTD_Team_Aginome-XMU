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
parser.add_argument("-In", type=str, help="Input expression file (with genes specified using HUGO symbols)", default="./expr.csv")
parser.add_argument("-Out", type=str, help="Output result file", default="./predicton.csv")
parser.add_argument("-scale", type=str, help="The scale of the expression data (i.e., Log2, Log10, Linear)", default="Linear")
parser.add_argument("-model", type=str, help="Deep-learing model file trained by DAISM", default="./output/DAISM_model.pkl")
parser.add_argument("-dataset", type=str, help="name of test dataset", default="test")
parser.add_argument("-celltype", type=str, help="Model celltypes", default="./output/DAISM_model_celltypes.txt")
parser.add_argument("-feature", type=str, help="Model feature", default="./output/DAISM_model_feature.txt")



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

class MLP(torch.nn.Module):  
    def __init__(self,INPUT_SIZE,OUTPUT_SIZE):
        super(MLP, self).__init__() 
        # Architectures 
        L1 = 1024
        L2 = 512
        L3 = 256
        self.hidden = torch.nn.Sequential(                       
            nn.Linear(INPUT_SIZE, L1),
            nn.ReLU(),
            nn.Linear(L1,L2),
            nn.BatchNorm1d(L2),
            nn.ReLU(),
            nn.Linear(L2,L3),
            nn.ReLU(),
        )
        self.predict =  torch.nn.Sequential( 
            nn.Linear(L3, OUTPUT_SIZE),
        )
    def forward(self, x):   
        y = self.hidden(x)    
        y = self.predict(y)   
        return y

def model_load(feature, celltypes, modelpath):
    """
    Load trained model
    :param feature:
    :param celltypes:
    :param modelpath:
    :return:
    """
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # Initialize model
    model = MLP(INPUT_SIZE = len(feature),OUTPUT_SIZE = len(celltypes)).double()

    # Load trained model
    model.load_state_dict(torch.load(modelpath, map_location='cpu'))
        
    return model    
    
    
def do_MLP(expression_paths,scale,model,dataset_name,feature,celltypes):
    exp = pd.read_csv(expression_paths, sep=",", index_col=0)
    print('Load data!')
    if scale == "Log2":
        exp = pow(2,exp)

    if scale == "Log10":
        exp = pow(10,exp)
        
    if any(exp<0):
        exp = exp + abs(exp.min().min())
    
    exp_data = exp.reindex(feature).values
            
    out = MCpred(model=model,data=exp_data)
            
    pred = Variable(out,requires_grad=False).cpu().numpy().reshape(exp_data.shape[1],len(celltypes))

    pred_result = pd.DataFrame(pred.T,index=celltypes,columns=exp.columns)
    
    pred_result['cell.type']=pred_result.index
    result = pd.melt(pred_result,id_vars='cell.type',var_name='sample.id',value_name='prediction')
    result['dataset.name']=dataset_name
    
    return result    
    

if __name__ == '__main__':
    
    ############################
    #### prediction modules ####
    ############################
    
    inputArgs = parser.parse_args()
    
    # Load signature genes and celltype labels
    feature = pd.read_csv(inputArgs.feature,sep='\t')['0']
    celltypes = pd.read_csv(inputArgs.celltype,sep='\t')['0']

    # Load trained model
    model = model_load(feature, celltypes, inputArgs.model)

    # Prediction
    result = do_MLP(inputArgs.In, inputArgs.scale, model, inputArgs.dataset, feature, celltypes)
    
    # Save predicted result
    if os.path.exists(inputArgs.Out)==False:
        os.mkdir(inputArgs.Out)
        
    result.to_csv(inputArgs.Out, sep=',', index=False)

    print('Prediction finished!')
    