# HybridGCN
The source code for HybridGCN for Protein Solubility Prediction with Adaptive Weighting of Multiple Features (https://jcheminf.biomedcentral.com/articles/10.1186/s13321-023-00788-8)

## 1. Dependencies
The code needs the installation of ESM (https://github.com/facebookresearch/esm) for esm-1v feature extraction. 

The code has been tested under Python 3.7.9, with the following packages installed (along with their dependencies):
- torch==1.6.0
- numpy==1.19.1
- scikit-learn==0.23.2
- pandas==1.1.0
- tqdm==4.48.2

## 2. Extract the ESM-1v features
- Install ESM for esm-1v feature extraction.To install ESM, following the introduction on https://github.com/facebookresearch/esm.
- Extract ESM-1v features for the training and test sets using esm_features_extraction.py.

## 3. Retrain and test the HybridGCN model
### Step 1: Download all sequence features
Please go to the path `./Data/Feature Link.txt` and download `Node Features.zip` and `Edge Features.zip`

### Step 2: Decompress all `.zip` files
Please unzip 3 zip files and put them into the corresponding paths.
- `./Data/node_features.zip` -> `./Data/node_features`
- `./Data/edge_features.zip` -> `./Data/edge_features`
- `./Data/fasta.zip` -> `./Data/fasta`

### Step 3: Run the training code
Run the following python script.
```
$ python Train_ESM.py
```
A trained model will be saved in the folder `./Model` and validation results in the folder `./Result`

### Step 4: Run the test code
Run the following python script and it will be finished in a few seconds.
```
$ python Test_ESM.py
```

## 6. Citations
Please cite our paper if you want to use our code in your work.
```bibtex
@article{chen2023hybridgcn,
  title={HybridGCN for Protein Solubility Prediction with Adaptive Weighting of Multiple Features},
  author={Chen, Long and Wu, Rining and Zhou, Feixiang and Zhang, Huifeng and Liu, Jian K},
  journal={Journal of cheminformatics},
  year={2023},
  publisher={Springer}
}
