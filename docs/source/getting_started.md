# Getting started

you need to download `best_mtp_moc_vfp.ckpt` that is the pretrained model with MTP + MOC + VFP tasks from [figshare](https://figshare.com/articles/dataset/MOFTransformer/21155506).

you can change parameters `model/config.py` for pre-training, fine-tuning, and test.


## 1. Training
Before training, you should check the environment of server you want to run and modify the `my_env` function in `conf.py`.
Given `MOFTransformer` have over 85 M of parameters, we strongly recommend running with GPUs.

```python
def my_env():
    num_gpus = 1 # the number of GPUs
    num_nodes = 1 # the number of server
    num_workers = 16  # the number of cpu's core

    data_root = "examples/dataset"
    log_dir = "examples/logs"
```

### Fine-tuning
 In order to load the wights of pre-trained model as initial weights, you should change `load_path` in `conf.py`.
The following script will fine-tune the pre-model with the `examples` directory.

```python
run.py with downstream_example my_env
```
you can find the parameter for `downstream_example` in `config.py`

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

if you download `downstream_release.tar.gz` in [figshare](https://figshare.com/articles/dataset/MOFTransformer/21155506), 
you can run the fine-tuning examples for 

### Pre-training
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

```python
run.py with task_mtp my_env
```

if you need the 1M hMOF created by `PORMAKE`, please contact us by email. 

## 2. Test

## 3. Visualization