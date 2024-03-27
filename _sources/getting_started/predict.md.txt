# Predict

To obtain the output of a model on a dataset, one can utilize the predict function, which conveniently saves the result in CSV format at the location of the current model.

## 1. Using `predict` function

You can train the model using the function `predict` in `moftransformer` python library.
> **moftransformer.predict** (root_dataset, load_path, downstream=None, split='all', save_dir='None, **kwargs)

- root_dataset : A folder containing graph data, grid data, and json of MOFs that you want to train or test.
  (see [**generate custom dataset**](https://hspark1212.github.io/MOFTransformer/dataset.html#generate-custom-dataset))
- load_path: Path for model you want to load and predict (*.ckpt).  
- downstream: Name of user-specific task (e.g. bandgap, gasuptake, etc).
- split: The split you want to predict on your dataset ('all', 'train', 'test', or 'val')
- save_dir: Path for directory you want to save *.csv file. (default : None -> path for loaded model)
- kwargs : configuration for MOFTransformer

## Example for predict using python
```python
from pathlib import Path
import moftransformer
from moftransformer.examples import example_path

root_dataset = example_path['root_dataset']
downstream = example_path['downstream']

# Get ckpt file
seed = 0               # default seeds
version = 0            # version for model. It increases with the number of trains
checkpoint = 'best'    # Epochs where the model is stored. 
mean = 0
std = 1

load_path = Path(log_dir) / f'pretrained_mof_seed{seed}_from_pmtransformer/version_{version}/checkpoints/{checkpoint}.ckpt'

if not load_path.exists():
    raise ValueError(f'load_path does not exists. check path for .ckpt file : {load_path}')
    
moftransformer.predict(
    root_dataset, load_path=load_path, downstream=downstream, split='all', mean=mean, std=std
)
```

## 2. Using command-line

You can proceed with prediction in the `command-line` using parameters the same as Python's predict.

```bash
$ moftransformer predict --root_dataset [root_dataset] --load_path [load_path]--downstream [downstream] --split [split] --save_dir [save_dir] ...
```

For example:
```bash
$ moftransformer predict --root_dataset './data' --load_path 'path/of/model' --downstream 'exmaple' --mean 0. --std 1.
```

For more information, see the help, command by "moftransformer predict -h".