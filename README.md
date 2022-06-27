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
torch 
numpy
pandas
scikit-learn
argh
```
3. Enter the workspace.

```
cd DCTD_Team_Aginome-XMU
```

4. Download pretrained models from [here](https://doi.org/10.17632/yt79wsksg9.1) and save to ```DCTD_Team_Aginome-XMU``` folder.

Coarse-grained:

The Challenge consisted of a “coarse-grained” sub-Challenge, during which participants predicted levels of eight major immune and stromal cell populations [including B cells, CD4+ and CD8+ T cells, NK cells, neutrophils, cells within the monocytic lineage (monocytes/macrophages/dendritic cells), endothelial cells, and fibroblasts].

Fine-grained:

The Challenge also consisted of a “fine-grained” sub-Challenge, during which participants further dissected major populations into 14 minor sub-populations according to their functional orientation (e.g., naive B cells, memory B cells, naive CD4 T cells, memory CD4 T cells, naive.CD8 T cells, memory CD8 T cells, regulatory T cells, monocytes, macrophages, myeloid dendritic cells, NK cells, neutrophils, endothelial cells, and fibroblasts).
5. Extract tar.gz file.
```
tar -zxvf coarse_models.tar.gz
tar -zxvf fine_models.tar.gz
```
The pretrained models of coarse-grained and fine-grained are saved in sub-folders ```coarse_models``` and ```fine_models``` respectively.


**Note:** It takes 10-20 minutes to prepare the environment. Downloading pre-trained models take a long time.
### Usage on demo
```
python run_DCTD.py 

    positional arguments:
    {coarse,fine}  Select one of the following sub-commands
        coarse       coarse-grained deconvolution
        fine         fine-grained deconvolution
    
    optional arguments:
        -In IN            Input expression file (with genes specified using HUGO symbols)
        -Out OUT          Output result file
        -scale SCALE      The scale of the expression data (i.e., Log2, Log10, Linear)
        -model MODEL      Trained models directory
        -dataset DATASET  name of test dataset
```
### Input
The input expression data should be a comma separated file with columns associated to sample ID and rows to genes specified using HUGO symbols.

| Gene      | S1 | S2     |...|
   | :----:        |    :----:   |          :---: |:---:|
   |A1BG|0.0038788036146706045|0.0054788910267111225|...|
   |A1BG-AS1|0.0021641861287775527|0.001029015600687667|...|
   |...|...|...|...|
### Run a demo

Use the following command for coarse-grained deconvolution:
   ```
   python run_DCTD.py coarse -In demo_data.csv -Out ./prediction.csv -scale Linear -model ./coarse_models/ -dataset demo_data
   ```

   Or you can use following command for fine-grained deconvolution:

   ```
   python run_DCTD.py fine -In demo_data.csv -Out ./prediction.csv -scale Linear -model ./fine_model/ -dataset demo_data
   ```

   **Note:** It takes only a few seconds to perform cell type proportion prediction.
### Expected outcomes

The output file ```prediction.csv``` is a comma separated file with 4 columns (cell type, sample ID, predicted proportion and dataset name).

   | cell.type      | sample.id | prediction     |dataset.name|
   | :----:        |    :----:   |          :---: |:---:|
   |CD4.T.cells|S1|0.20734058036020336|demo_data|
   |CD8.T.cells|S1|0.10328292389874755|demo_data|
   |NK.cells|S1|0.0654560242324504|demo_data|
   |...|...|...|demo_data|

#### Note

The coarse-grained and fine-grained models were trained on a gene set that contains 5080 genes. Please check whether these genes exist in your expression profile before performing prediction. If not, we recommend to use DAISM-DNN<sup>XMBD</sup> package (https://github.com/xmuyulab/DAISM-XMBD.git) to perform deconvolution by training models from scratch.
### Citation
Lin Y, Li H, Xiao X, et al. DAISM-DNN<sup>XMBD</sup>: Highly accurate cell type proportion estimation with in silico data augmentation and deep neural networks. Patterns (2022) https://doi.org/10.1016/j.patter.2022.100440