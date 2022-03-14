# DCTD_Team_Aginome-XMU
Team Aginome-XMU submissions to DREAM Challenge Tumor Deconvolution

### Installation
1. Clone the GitHub repo.
```
git clone https://github.com/xmuyulab/DCTD_Team_Aginome-XMU.git
```   
2. Install the following requirements.
```
python >= 3.6.0
numpy
pandas
scikit-learn
argparse
```
3. Enter the workspace.
```
cd DCTD_Team_Aginome-XMU
```
### Usage

1. Download trained models and save to ```./model/``` folder.
    #### Coarse-grained
    The trained models for round 1 can be downloaded from: https://figshare.com/s/4058d860f7fbf9b89cd2

    The Challenge consisted of a “coarse-grained” sub-Challenge, during which participants predicted levels of eight major immune and stromal cell populations [including B cells, CD4+ and CD8+ T cells, NK cells, neutrophils, cells within the monocytic lineage (monocytes/macrophages/dendritic cells), endothelial cells, and fibroblasts].
    #### Fine-grained
    The trained models for round 1 can be downloaded from: https://figshare.com/s/abe52254e69b8cc27ccc

    The Challenge also consisted of a “fine-grained” sub-Challenge, during which participants further dissected major populations into 14 minor sub-populations according to their functional orientation (e.g., naive B cells, memory B cells, naive CD4 T cells, memory CD4 T cells, naive.CD8 T cells, memory CD8 T cells, regulatory T cells, monocytes, macrophages, myeloid dendritic cells, NK cells, neutrophils, endothelial cells, and fibroblasts).

2. Use the following command for coarse-grained deconvolution:
```
python run_DCTD.py coarse -In expr.csv -Out ./prediction.csv -scale Linear -model ./model/ -dataset test
```
    Or you can use following command for fine-grained deconvolution:
```
python run_DCTD.py fine -In expr.csv -Out ./prediction.csv -scale Linear -model ./model/ -dataset test
```
#### Note
1. The expression data should be a comma separated file with columns associated to sample ID and rows to genes specified using HUGO symbols.
2. The coarse-grained and fine-grained models were trained on a gene set that contains 5080 genes. Please check whether these genes exist in your expression profile before performing prediction. If not, we recommend to use DAISM-DNN<sup>XMBD</sup> package to perform deconvolution by training models from scratch.
### Citation
Lin Y, Li H, Xiao X, et al. DAISM-DNN<sup>XMBD</sup>: Highly accurate cell type proportion estimation with in silico data augmentation and deep neural networks. Patterns (2022) https://doi.org/10.1016/j.patter.2022.100440