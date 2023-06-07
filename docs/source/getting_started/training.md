# Training

You can train MOFTransformers by the following methods.

After downloading the [**pre-trained model**](https://hspark1212.github.io/MOFTransformer/installation.html#download-model-and-data), it is possible to train a model with higher accuracy.

```bash
$ moftransformer download pretrain_model
```

## 1. Using `run` function

You can train the model using the function `run` in `moftransformer` python library.
> **moftransformer.run** (root_dataset, downstream, log_dir='logs/', test_only=False, **kwargs)

- root_dataset : A folder containing graph data, grid data, and json of MOFs that you want to train or test.
  (see [**generate custom dataset**](https://hspark1212.github.io/MOFTransformer/dataset.html#generate-custom-dataset))
- downstream: Name of user-specific task (e.g. bandgap, gasuptake, etc).
- log_dir: Directory to save log, models, and params. (default:`logs/`)
- test_only: If True, only the test process is performed without the learning model. (default:`False`)
- kwargs : configuration for MOFTransformer

** After version 2.1.1, the best model is saved separately as `best.ckpt`.


### Example for training PMTransformer (MOFTransformer):
```python
import moftransformer
from moftransformer.examples import example_path

# data root and downstream from example
root_dataset = example_path['root_dataset']
downstream = example_path['downstream']
log_dir = './logs/'
# load_path = "pmtransformer" (default)

# kwargs (optional)
max_epochs = 10
batch_size = 8
mean = 0
std = 1


moftransformer.run(root_dataset, downstream, log_dir=log_dir,                   
                   max_epochs=max_epochs, batch_size=batch_size,
                   mean=mean, std=std)
```

After training, the trained model, logs and hyperparameters will be saved at `log_dir`.  
Then you look over the results with tensorboard

```bash
$ tensorboard --logdir=[log_dir] --bind_all
```



## 2. Using command-line

You can proceed with training in the `command-line` using parameters the same as Python's run.

```bash
$ moftransformer run --root_dataset [root_dataset] --downstream [downstream] --logdir [logdir] ...
```

For example:
```bash
$ moftransformer run --root_dataset './data' --downstream 'example' --max_epochs 10 --devices 2 --batch_size 216
```

For more information, see the help, command by "moftransformer run -h".