# Dataset Preparation

## 1. Introdction

`MOFTransformer` takes both atom-wise graph embeddings and energy-grid embeddings to capture local and global features,
respectively.

(1) atom-wise graph embeddings

Tha atom-wise graph embeddings are taken from the modified [CGCNN](https://github.com/txie-93/cgcnn.git) by removing pooling layer and adding topologically unique
atom selection.

(2) energy-grid embeddings

The 3D energy grids are calculated by [GRIDDAY](https://github.com/Sangwon91/GRIDAY.git) with the united atom model of
methane molecule using UFF.

## 2.Generate custom dataset

From cif files, `moftransformer/utils/prepare_data.py` file will generate inputs of MOFTranformer which are the atom-wise graph embeddings and
enery-grid embeddings.
You need to prepare `cif files (structures)` and `json files (targets ex. property, class)]` in `root_cifs` directory.

### randomly split dataset
If name of the json file is `raw_{task}.json`, then it will be randomly splitted by 8:1:1 (train:val:test). 

### custom splitted dataset
If you want to split data yourself, you just manually make splitted json files (train.json, val.json, test.json) at `root_cifs`.

```python
from moftransformer.utils.prepare_data import prepare_data
prepare_data(root_cifs, root_dataset, task="example") 
```

The example of json files is as follows.

```
{ 
    cif_id : property (float) or classes (int),
    ...
}
```

### 1. randomly split dataset
The example of `root_cifs` directory is as follows.

    root_cifs # root for cif files
    ├── [cif_id].cif
    ├── ...
    └── raw_{task}.json

### 2. custom splitted dataset
The example of `root_cifs` directory is as follows.

    root_cifs # root for cif files
    ├── [cif_id].cif
    ├── ...
    ├── train_{task}.json
    ├── val_{task}.json
    └── test_{task}.json

Then, You need to set parameters `root_dataset`, `task`.
`root_dataset`: the saved directories of input files
`task` : name of user-specific task (e.g. band gap, gas uptake, etc).

Finally, `prepare_data.py` will generate the atom-wise graph embeddings and energy-grid embeddings in `root_dataset`
directory.

    root_dataset # root for generated inputs 
    ├── train
    │   ├── [cif_id].graphdata # graphdata
    │   ├── [cif_id].grid # energy grid information
    │   ├── [cif_id].griddata16 # grid data
    │   ├── [cif_id].cif # primitive cif
    │   └── ...
    ├── val
    │   ├── [cif_id].graphdata # graphdata
    │   ├── [cif_id].grid # energy grid information
    │   ├── [cif_id].griddata16 # grid data
    │   ├── [cif_id].cif # primitive cif
    │   └── ...
    ├── test    
    │   ├── [cif_id].graphdata # graphdata
    │   ├── [cif_id].grid # energy grid information
    │   ├── [cif_id].griddata16 # grid data
    │   ├── [cif_id].cif # primitive cif
    │   └── ...
    ├── train_{task}.json
    ├── val_{task}.json
    └── test_{task}.json

## 3. Dataset for public database (CoREMOF, QMOF).

we've provided the dataset of atom-wise graph embedding and energy-grid embedding for the CoREMOF and the QMOF database
in our [**figshare**](https://figshare.com/articles/dataset/MOFTransformer/21155506) database.

