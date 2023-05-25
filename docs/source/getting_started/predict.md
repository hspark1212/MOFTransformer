# Predict

To obtain the output of a model on a dataset, one can utilize the predict function, which conveniently saves the result in CSV format at the location of the current model.

## Example for predict using python
```python
from pathlib import Path
import moftransformer
from moftransformer.examples import example_path

root_dataset = example_path['root_dataset']
downstream = example_path['downstream']

# Get ckpt file
log_dir = './logs/'    # same directory make from training
seed = 0               # default seeds
version = 0            # version for model. It increases with the number of trains
checkpoint = 'last'    # Epochs where the model is stored. 

load_path = Path(log_dir) / f'pretrained_mof_seed{seed}_from_pmtransformer/version_{version}/checkpoints/{checkpoint}.ckpt'

if not load_path.exists():
    raise ValueError(f'load_path does not exists. check path for .ckpt file : {load_path}')

moftransformer.predict(
    root_dataset, load_path=load_path, downstream=downstream, split='all'
)
```