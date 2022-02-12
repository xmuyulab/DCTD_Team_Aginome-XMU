# package
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import numpy as np
from sklearn import preprocessing as pp
import pandas as pd
import os


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

class MLP1(torch.nn.Module):  
    def __init__(self,INPUT_SIZE,OUTPUT_SIZE):
        super(MLP1, self).__init__()     
        
        L1 = 1024
        L2 = 512
        L3 = 256
        L4 = 128
        L5 = 32
        self.hidden = torch.nn.Sequential(                       
            nn.Linear(INPUT_SIZE, L1),
            # nn.Dropout(0.1),
            nn.Tanh(),
            nn.Linear(L1,L2),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(L2,L3),
            #nn.Dropout(0.1),
            nn.Tanh(),
            nn.Linear(L3,L4),
            #nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(L4,L5),
            nn.Tanh(),
        )
        self.predict =  torch.nn.Sequential( 
            nn.Linear(L5, OUTPUT_SIZE),
            # nn.Softmax()
        )
    def forward(self, x):   
        
        y = self.hidden(x)     
        y = self.predict(y)    
        return y
class MLP2(torch.nn.Module):  
    def __init__(self,INPUT_SIZE,OUTPUT_SIZE):
        super(MLP2, self).__init__()     
        
        L1 = 1024
        L2 = 1024
        L3 = 512
        L4 = 256
        L5 = 32
        self.hidden = torch.nn.Sequential(                       
            nn.Linear(INPUT_SIZE, L1),
            # nn.Dropout(0.1),
            nn.Tanh(),
            nn.Linear(L1,L2),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(L2,L3),
            #nn.Dropout(0.1),
            nn.Tanh(),
            nn.Linear(L3,L4),
            #nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(L4,L5),
            nn.Tanh(),
        )
        self.predict =  torch.nn.Sequential( 
            nn.Linear(L5, OUTPUT_SIZE),
            # nn.Softmax()
        )
    def forward(self, x):   
        
        y = self.hidden(x)    
        y = self.predict(y)    
        return y  

def do_MLP(dataset_names,expression_paths,scale):
    exp = pd.read_csv(expression_paths, sep=",", index_col=0)
    
    if scale == "Log2":
        exp = pow(2,exp)
        
    if any(exp<0):
        exp = exp + abs(exp.min().min())
    
    exp_data = exp.reindex(sig_genes).values


    final_result = np.zeros(shape=(14,exp_data.shape[1]))
    modelpath = 'model/'
    for c, celltype in enumerate(celltypes):
        file_list = os.listdir(modelpath+celltype)
        for i,file in enumerate(file_list):
            if c == 4 or c ==5 or c ==6 or c ==7:
                model1.load_state_dict(torch.load(modelpath+celltype+'/'+file,map_location='cpu'))
                model = model1 
            else:
                model2.load_state_dict(torch.load(modelpath+celltype+'/'+file,map_location='cpu'))
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
    result['dataset.name']=dataset_names
    
    return result

input_df = pd.read_csv("input/input.csv",sep=',')
dataset_names = input_df['dataset.name']
expression_files = input_df['hugo.expr.file']
expression_paths = "input/"+expression_files
scale = input_df['scale']

sig_genes = pd.read_csv('validation_features_5080.txt',sep="\t",index_col=0)
sig_genes= list(sig_genes['feature_left'])

celltypes = ['naive.B.cells', 'memory.B.cells', 'naive.CD4.T.cells','memory.CD4.T.cells','regulatory.T.cells', 'naive.CD8.T.cells','memory.CD8.T.cells', 'NK.cells', 'monocytes','myeloid.dendritic.cells', 'macrophages', 'neutrophils', 'fibroblasts','endothelial.cells']

torch.manual_seed(0)
model1 = MLP1(INPUT_SIZE=len(sig_genes),OUTPUT_SIZE=len(celltypes)).double()
model2 = MLP2(INPUT_SIZE=len(sig_genes),OUTPUT_SIZE=len(celltypes)).double()

combined_result_df= pd.DataFrame(columns = ['cell.type','sample.id','prediction','dataset.name'])

for i in range(len(dataset_names)):
    result = do_MLP(dataset_names[i],expression_paths[i],scale[i])
    combined_result_df = combined_result_df.append(result)
    
order = ['dataset.name','sample.id','cell.type','prediction']
combined_result_df = combined_result_df[order] 

if os.path.exists("output/")==False:
    os.mkdir("output/")

combined_result_df.to_csv("output/predictions.csv",sep=',', index=False)
