# Test

When you want to check the test after the training is finished, you can proceed with the test through the argument `test_only` and `load_path`.

## Example for test using python
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

load_path = Path(log_dir) / f'pretrained_mof_seed{seed}_from_/version_{version}/checkpoints/{checkpoint}.ckpt'

if not load_path.exists():
    raise ValueError(f'load_path does not exists. check path for .ckpt file : {load_path}')

moftransformer.run(root_dataset, downstream, test_only=True,
                   load_path=load_path)
```


## Example for test using command-line
It can be executed using the command line in the same way.
```bash
$ moftransformer run --root_dataset './data' --downstream 'exmaple' --test-only True --config load_path='path_load'
```