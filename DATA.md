# Dataset Preparation
## 1. Introdction

`MOFformer` takes both CGCNN w/o pooling layer and 1D flatten patches of 3D energy grid as inputs.

(1) CGCNN W/O pooling layer

We modified CGCNN code (https://github.com/txie-93/cgcnn.git) by removing pooling layer and adding topologically unique atom selection. .
(unique atoms are topologically same atoms, meaning they are same to 3th edges in graph)

(2) 1D flatten patches of 3D energy grid

The 3D energy grid are calculated by GRIDDAY (https://github.com/Sangwon91/GRIDAY.git) with unique atom model of methane molecule using UFF.

 
## 2.Generate dataset
It is required that `cif files (structures)` and `json files (targets ex. property, class)]` are placed in `root_cifs` directory,
which were splited into `train`,`val` and `test`(optional). 

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
    ├── targe_train.json
    ├── targe_val.json
    └── target_test.json

Then, please use `model/utils/prepare_data.py` to generate dataset.

```angular2html
from model.utils.prepare_data import prepare_data
prepare_data(root_cifs, root_dataset) 
```


Finally, `prepare_data.py` will generate crystal graph and energy grid in `root_dataset` directory.

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
    ├── test (optional)      
    │   ├── [cif_id].graphdata # graphdata
    │   ├── [cif_id].grid # energy grid information
    │   ├── [cif_id].griddata16 # grid data
    │   ├── [cif_id].cif # primitive cif
    │   └── ...
    ├── target_train.json
    ├── target_val.json
    └── target_test.json (optional)


