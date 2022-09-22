# Dataset Preparation
## 1. Introdction

`MOFTransformer` takes both atom-wise graph embeddings and energy-grid embeddings to capture local and global features, respectively. 

(1) atom-wise graph embeddings

We modified CGCNN code (https://github.com/txie-93/cgcnn.git) by removing pooling layer and adding topologically unique atom selection. .
(unique atoms are topologically same atoms, meaning they are same to 3th edges in graph)

(2) energy-grid embeddings

The 3D energy grid are calculated by GRIDDAY (https://github.com/Sangwon91/GRIDAY.git) with the united atom model of methane molecule using UFF.

 
## 2.Generate custom dataset
 From cif files, `model/utils/prepare_data.py` file will generate the inputs which are the atom-wise graph embeddings and enery-grid embeddings.
You need to prepare `cif files (structures)` and `json files (targets ex. property, class)]` in `root_cifs` directory.
The json files should be splited into `train`,`val` and `test`. 

You can find an example of generating custom datasets in `demo.ipynb` and `examples` directory.
```python
from model.utils.prepare_data import prepare_data
prepare_data(root_cifs, root_dataset, task="example") 
```

The example  of json files is as follows.
```
{ 
    cif_id : property (float) or classes (int),
    ...
}
```
The example of `root_cifs` directory is as follows.

    root_cifs # root for cif files
    ├── [cif_id].cif
    ├── ...
    ├── train_{task}.json
    ├── val_{task}.json
    └── test_{task}.json

Then, You need to set parameters `root_dataset`, `task`.
`root_dataset`: the saved directories of input files 
`task` : name of user-specific task (e.g. bandgap, gasuptake, etc).

Finally, `prepare_data.py` will generate the atom-wise graph embeddings and energy-grid embeddings in `root_dataset` directory.

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
    ├── target_{task}.json
    └── target_{task}.json

## 3. Dataset for public database (CoREMOF, QMOF).
we've provided the dataset of atom-wise graph embedding and energy-grid embedding for the CoREMOF and the QMOF database in our [**figshare**](https://figshare.com/articles/dataset/MOFTransformer/21155506) database.

