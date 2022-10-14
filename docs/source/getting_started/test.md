# Test

After training, the trained model, logs and hyperparameters will be saved at `log_dir` in `config.py`.
Then you look over the results with `tensorboard`

```shell
tensorboard --logdir=examples/logs --bind_all
```

You can calculate the scores for test set.
you set the parameters `test_only=True` and `load_path`

```shell
moftransformer run downstream_raspa_100bar load_path={ckpt path of fine-tuning model} test_only=True
```