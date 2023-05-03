# Prepare data

It provides a `prepare_data` function that allows to generate input data of `MOFTransformer` (i.e., atom-wise graph embeddings and energy-grids) 
The details are explained in the [Data Preparation](https://hspark1212.github.io/MOFTransformer/dataset.html) section.

In order to run `prepare_data`, you need to install `GRIDAY` to calculate energy-grids from cif files.

You can download GRIDAY using [command-line](https://hspark1212.github.io/MOFTransformer/installation.html#installation-using-command-line)
or [python](https://hspark1212.github.io/MOFTransformer/installation.html#installation-using-python)

```bash
$ moftransformer install-griday
```

## Example for running `prepare-data`

As an example, you can run `prepare_data` with 10 cif files in `moftransformer/examples/raw` directory
(For more information, see [Dataset Preparation](https://hspark1212.github.io/MOFTransformer/dataset.html))

```python
from moftransformer.examples import example_path
from moftransformer.utils import prepare_data

# Get example path
root_cifs = example_path['root_cif']
root_dataset = example_path['root_dataset']
downstream = example_path['downstream']

train_fraction = 0.7  # default value : 0.8
test_fraction = 0.2   # default value : 0.1

# Run prepare data
prepare_data(root_cifs, root_dataset, downstream=downstream, 
             train_fraciton=train_fraction, test_fraciton=test_fraction)
```
