# Dataset Preparation
(test)
## 1. Introdction
crystal graph and energy grid are uesd for dataset of `pretrained_mof`.

(1) crystal graphs include information of neighbors (nbr) such as distance, atomic numbers, unique atoms.
(unique atoms are topologically same atoms, meaning they are same to 3th edges in graph)

(2) energy grid are calculated by GRIDDAY (Energy shape calculator for the porous materials, https://github.com/Sangwon91/GRIDAY.git )
 
## 2.Generate dataset
So, it is required that `cif files (structures)` and `json files (targets ex. property, class)]` are placed in `root_cifs` directory, which were splited into `train`,`val` and `test`(optional). 

The example  of json files is as follows.
```

{ 
    cif_id : property (float) or classes (int),
    ...
}
```
The example of `root_cifs` directory is as follows.

    root_cifs # root for cif files
    ├── train            
    │   ├── [cif_id].cif
    │   ├── ...
    │   └── target_train.json
    ├── val       
    │   ├── [cif_id].cif
    │   ├── ...
    │   └── target_val.json
    └── test (optional)
        ├── [cif_id].cif
        ├── ...
        └── target_test.json
    

Then, please use `model/utils/prepare_data.py` to generate dataset.
```angular2html
from model.utils.prepare_data import prepare_data
prepare_data(root_cifs, root_dataset) 
```
Finally, `prepare_data.py` will generate crystal graph and energy grid in `root_dataset` directory.

    root_dataset # root for generated inputs 
    ├── train            
    │   ├── [cif_id].grid # energy grid information
    │   ├── [cif_id].griddata # grid data
    │   ├── ...
    ├── val          
    │   ├── [cif_id].grid # energy grid information
    │   ├── [cif_id].griddata # grid data
    │   ├── ...
    ├── test (optional)      
    │   ├── [cif_id].grid # energy grid information
    │   ├── [cif_id].griddata # grid data
    │   ├── ...
    ├── train.arrow
    ├── val.arrow
    └── test.arrow (optional)


