# Training

In `model/config.py`, you can change setting parameters for training, test, visualization.

Before training, you should check the environment of server you will run and modify the `my_env` function in `conf.py`.
Unfortunately, `MOFTransformer` is not available with CPUs. Given `MOFTransformer`have over 85 M of parameters, we
strongly recommend to use the server containing with GPUs.

```python
def my_env():
    num_gpus = 1 # the number of GPUs
    num_nodes = 1 # the number of server
    num_workers = 16  # the number of cpu's core

    data_root = "examples/dataset"
    log_dir = "examples/logs"
```

## Fine-tuning

In oder to fine-tuning the pre-trained `MOFTransforemr`, you should download `best_mtp_moc_vfp.ckpt` that is the
pretrained model with MTP + MOC + VFP tasks from [**
figshare**](https://figshare.com/articles/dataset/MOFTransformer/21155506) for the first time.
Then, you should set `load_path` in `conf.py` to load the wights of pre-trained model as initial weights.

### Fine-tuning with downstream dataset

if you download `downstream_release.tar.gz` in [**
figshare**](https://figshare.com/articles/dataset/MOFTransformer/21155506),
you can run the fine-tuning examples for H<sub>2</sub> uptake (`raspa_100bar`) and dilute diffusivity in log
scale (`diffusivity_log`)

```python
@ex.named_config
def downstream_raspa_100bar():
    exp_name = "downstream_raspa_100bar"
    data_root = "examples/downstream_release"
    log_dir = "examples/logs"
    downstream = "raspa_100bar"
    load_path = "examples/best_mtp_moc_vfp.ckpt"  # should be set

    # trainer
    max_epochs = 20
    batch_size = 32
    per_gpu_batchsize = 8

    # model
    loss_names = _loss_names({"regression": 1})

    # normalize
    mean = 487.841
    std = 63.088
```

```python
def downstream_diffusivity_log():
    exp_name = "downstream_diffusivity_log"
    data_root = "examples/downstream_release"
    log_dir = "examples/logs"
    downstream = "diffusivity_log"
    load_path = "examples/best_mtp_moc_vfp.ckpt"  # should be set

    # trainer
    max_epochs = 20
    batch_size = 32
    per_gpu_batchsize = 8

    # model
    loss_names = _loss_names({"regression": 1})

    # normalize
    mean = -8.300
    std = 1.484
```

Then, you can run the fine-tuning examples:

```shell
run.py with  downstream_raspa_100bar my_env
```

or

```shell
run.py with  downstream_raspa_100bar my_env
```

### Fine-tuning with custom dataset

Here is an example for fine-tuning with `example/datasets`

```python
def downstream_example():
    exp_name = "downstream_example"
    data_root = "examples/dataset"
    log_dir = "examples/logs"
    downstream = "example"
    load_path = "examples/best_mtp_moc_vfp.ckpt"  # should be set

    # trainer
    max_epochs = 20
    batch_size = 32
    per_gpu_batchsize = 8

    # model
    loss_names = _loss_names({"regression": 1})

    # normalize
    mean = None
    std = None
```

Then, you can run the fine-tuning examples:

```shell
run.py with downstream_example my_env
```

## Pre-training

Although we already provide `best_mtp_moc_vfp.ckpt` pretrained with 1M hMOF, you can also pre-train your mdoel.
There are 6 types of pretraining tasks. (i.e. ggm, mpp, mtp, vfp, moc, bbp)

```python
def _loss_names(d):
    ret = {
        "ggm": 0,  # graph grid matching
        "mpp": 0,  # masked patch prediction
        "mtp": 0,  # mof topology prediction
        "vfp": 0,  # (accessible) void fraction prediction
        "moc": 0,  # metal organic classification
        "bbp": 0,  # building block prediction
        "classification": 0,  # classification -> fine-tuning 
        "regression": 0,  # regression -> fine-tuning
    }
    ret.update(d)
    return ret
```

```shell
run.py with task_mtp my_env
```

if you need the 1M hMOF created by `PORMAKE`, please contact us by email. 