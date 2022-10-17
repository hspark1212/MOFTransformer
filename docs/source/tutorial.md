# 1. Download pretrain_model and hMOF file
First, download pretrain_model
```bash
$ moftransformer download pretrain_model
```

Next, download [hMOF database](https://figshare.com/articles/dataset/MOFTransformer/21155506) to the current folder.
```bash
$ moftransformer download hmof -o ./hmof 
```

The configuration of the generated hMOF folder is as follows.

    hmof
    └── downstream_release
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
        │
        ├── train_diffusivity_log.json
        ├── val_diffusivity_log.json
        ├── test_diffusivity_log.json
        │
        ├── train_raspa_100bar.json
        ├── val_raspa_100bar.json
        └── test_raspa_100bar.json

Here, the calculated values for the two tasks (diffusivity, uptake) exist in the json file.

# 2. Train MOFTransformer
Training MOFTransformer is conducted based on the download hMOF database.

```python
import moftransformer

data_root = './hmof/downstream_release'
downstream = 'raspa_100bar'
log_dir = './logs'
max_epochs = 20
mean = 487.841
std = 63.088
batch_size = 32

moftransformer.run(data_root, downstream, max_epochs=max_epochs, mean=mean, std=std, 
                   batch_size=batch_size, log_dir=log_dir)
```
Trained model and their hyper-parameters are saved in `log_dir` folder.

# 3. Test MOFTransformer
In order to proceed with the test of the saved model, the `.ckpt` of the file must be loaded.


```python
import moftransformer

data_root = './hmof/downstream_release'
downstream = 'raspa_100bar'
load_path = './logs/pretrained_mof_seed0_from_pretrained_model/version_0/checkpoints/last.ckpt' 
             # cpkt : ./logs/[target_model_path]/[version]/checkpoints/[model].ckpt
max_epochs = 20
mean = 487.841
std = 63.088
batch_size = 32

moftransformer.run(data_root, downstream, test_only=True, load_path=load_path, 
                   max_epochs=max_epochs, mean=mean, std=std, batch_size=batch_size)
```