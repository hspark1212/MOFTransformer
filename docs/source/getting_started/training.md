# Training

You can train MOFTransformers by the following methods.

After downloading the [**pre-trained model**](https://hspark1212.github.io/MOFTransformer/installation.html#download-model-and-data), it is possible to train a model with higher accuracy.

```bash
$ moftransformer download pretrain-model
```

## 1. Using `run` function

You can train the model using the function `run` in `moftransformer` python library.
> **moftransformer.run** (data_root, downstream, log_dir='logs/', test_only=False, **kwargs)

- data_root : A folder containing graph data, grid data, and json of MOFs that you want to train or test.
  (see [**generate custom dataset**](https://hspark1212.github.io/MOFTransformer/dataset.html#generate-custom-dataset))
- downstream: Name of user-specific task (e.g. bandgap, gasuptake, etc).
- log_dir: Directory to save log, models, and params. (default:`logs/`)
- test_only: If True, only the test process is performed without the learning model. (default:`False`)
- kwargs : configuration for MOFTransformer


### Example for training MOFTransformer:
```python
import moftransformer
from moftransformer.examples import example_path

# data root and downstream from example
data_root = example_path['data_root']
downstream = example_path['downstream']
log_dir = './logs/'

# kwargs (optional)
max_epochs = 10
batch_size = 8

moftransformer.run(data_root, downstream, log_dir=log_dir, 
                   max_epochs=max_epochs, batch_size=batch_size,)
```

### Example for test
```python
import moftransformer
from moftransformer.examples import example_path
from moftransformer.database import DEFAULT_PRETRAIN_MODEL_PATH

data_root = example_path['data_root']
downstream = example_path['downstream']
load_path = DEFAULT_PRETRAIN_MODEL_PATH

moftransformer.run(data_root, downstream, test_only=True,
                   load_path=load_path)
```

After training, the trained model, logs and hyperparameters will be saved at log_dir in config.py. 
Then you look over the results with tensorboard
```bash
$ tensorboard --logdir=[log_dir] --bind_all
```



## 2. Using command-line

You can proceed with training in the `command-line` using parameters the same as Python's run.

```bash
$ moftransformer run --data_root [data_root] --downstream [downstream] --logdir [logdir] [--test_only] --config [configuration]  
```

In the `config` argument, it is used in the form of `parameter=value`, and several factors can be received at once.

For example:
```bash
$ moftransformer run --data_root './data' --downstream 'exmaple' --config max_epcohs=10 devices=2 batch_size=216
```