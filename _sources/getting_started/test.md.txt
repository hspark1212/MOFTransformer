# Test

When you want to check the test after the training is finished, you can proceed with the test using `test` function.

The result of the test is stored in `save_dir` (result.json). If no `save_dir` is specified, they are saved to the model's location (`load_path`).

## Example for test using python
```python
from pathlib import Path
import moftransformer
from moftransformer.examples import example_path

root_dataset = example_path['root_dataset']
downstream = example_path['downstream']

# Get ckpt file
seed = 0               # default seeds
version = 0            # version for model. It increases with the number of trains

# For version > 2.1.1, best.ckpt exists
checkpoint = 'best'    # Epochs where the model is stored. 
save_dir = 'result/'

# optional keyword
mean = 0
std = 1

load_path = Path(log_dir) / f'pretrained_mof_seed{seed}_from_pmtransformer/version_{version}/checkpoints/{checkpoint}.ckpt'

if not load_path.exists():
    raise ValueError(f'load_path does not exists. check path for .ckpt file : {load_path}')

moftransformer.test(root_dataset, load_path, downstream=downstream,
                   save_dir=save_dir, mean=mean, std=std)
```



## Using run function
If you set `test_only` to True in the `run` function, you can proceed with the test like using `test` function. Unlike the test function, the logger runs, which you can see in tensorboard. If you want to check the test results in tensorboard, you can use the following method.

```python
from pathlib import Path
import moftransformer
from moftransformer.examples import example_path

root_dataset = example_path['root_dataset']
downstream = example_path['downstream']

# Get ckpt file
seed = 0               # default seeds
version = 0            # version for model. It increases with the number of trains

# For version > 2.1.1, best.ckpt exists.
checkpoint = 'best'    # Epochs where the model is stored. 

# optional keyword
mean = 0
std = 1

load_path = Path(log_dir) / f'pretrained_mof_seed{seed}_from_pmtransformer/version_{version}/checkpoints/{checkpoint}.ckpt'

if not load_path.exists():
    raise ValueError(f'load_path does not exists. check path for .ckpt file : {load_path}')

moftransformer.run(root_dataset, downstream=downstream, log_dir='logs/',
                   test_only=True, load_path=load_path, 
                   mean=mean, std=std)
```

## Example for test using command-line
It can be executed using the command line in the same way.
```bash
$ moftransformer test --root_dataset './data' --downstream 'example' --load_path 'model.ckpt' --mean 0 --std 1
```