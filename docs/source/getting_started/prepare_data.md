# Prepare data


First, You should be download `GRIDAY` before run moftransformer.
You can download moftransformer using [command-line](https://hspark1212.github.io/MOFTransformer/installation.html#installation-using-command-line)
or [python](https://hspark1212.github.io/MOFTransformer/installation.html#installation-using-python)

```bash
$ moftransformer install-griday
```

## example for running `prepare-data`

To run MOFTransformer, The `.cif` data should be converted into `grid data` and `graph data`.
A code that pre-processing 10 example cifs is shown as an example. \
(For more information, see [Dataset Preparation](https://hspark1212.github.io/MOFTransformer/dataset.html))

```python
from moftransformer.examples import example_path
from moftransformer.utils import prepare_data

# Get example path
root_cifs = example_path['root_cif']
root_dataset = example_path['root_dataset']
downstream = example_path['downstream']

train_fraction = 0.7
test_fraction = 0.2

# Run prepare data
prepare_data(root_cifs, root_dataset, downstream=downstream, 
             train_fraciton=train_fraction, test_fraciton=test_fraction)
```
