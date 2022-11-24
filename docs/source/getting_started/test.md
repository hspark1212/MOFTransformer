# Test

When you want to check the test after the training is finished, you can proceed with the test through the argument `test_only` and `load_path`.

## Example for test using python
```python
import moftransformer
from moftransformer.examples import example_path

root_dataset = example_path['data_root']
downstream = example_path['downstream']
load_path = <fine_tuned_model_ckpt_file_saved_in_log_folder>

moftransformer.run(root_dataset, downstream, test_only=True,
                   load_path=load_path)
```


## Example for test using command-line
It can be executed using the command line in the same way.
```bash
$ moftransformer run --root_dataset './data' --downstream 'exmaple' --test-only True --config load_path='path_load'
```