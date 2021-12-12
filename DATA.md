# Dataset Preparation
jkfdjfkdjfk
Using pretrained_mof model, you just have only cif files.
From cif files, our code generate two types of inputs : crystal-graph and calculate energy grid.
Please place cif files in your `root_cifs` directory, which were splited into `train`,`val` and `test` (optional).

For finetune (regression, classification), please make json file for target and place it in `train`,`val` and `test`.

json format -> Dict(key: value);
key = filename, value = target

    root_cifs # root for cif files
    ├── train            
    │   ├── [cif_id].cif
    │   ├── [cif_id].cif
    │   ├── ...
    │   └── target_train.json
    ├── val       
    │   ├── [cif_id].cif
    │   ├── [cif_id].cif
    │   ├── ...
    │   └── target_val.json
    └── test (optional)
        ├── [cif_id].cif
        ├── [cif_id].cif
        ├── ...
        └── target_test.json
    

To generate inputs, please use `model/utils/prepare_data.py`.
```angular2html
from model.utils.prepare_data import prepare_data
prepare_data(root_cifs, root_dataset) 
```

    root_dataset # root for generated inputs 
    ├── train            
    │   ├── grid # energy grid information
    │   │   ├── [cif_id].grid
    │   │   └── ...
    │   ├── griddata # grid data
    │   │   ├── [cif_id].griddata
    │   │   └── ...
    ├── val          
    │   ├── grid
    │   │   ├── [cif_id].grid
    │   │   └── ...
    │   ├── griddata
    │   │   ├── [cif_id].griddata
    │   │   └── ...
    ├── test (optional)      
    │   ├── grid
    │   │   ├── [cif_id].grid
    │   │   └── ...
    │   ├── griddata
    │   │   ├── [cif_id].griddata
    │   │   └── ...
    ├── train.arrow
    ├── val.arrow
    └── test.arrow (optional)


