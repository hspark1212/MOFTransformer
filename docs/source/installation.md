# Installation

## 1. Requirements

- python >= 3.8 or newer
- Numpy


## 2. Installation

Given that `MOFTransformer` is based on pytorch, please install [pytorch](https://pytorch.org/get-started/locally/) (>= 1.12.0) according to your environments.

### Installation using PIP

The simplest way for installing `moftransformer` is to use PIP.

```bash
$ pip install moftransformer
```

### Editable install
If you want to modify the code, you can use the develop mode.

```bash
$ git clone https://github.com/hspark1212/MOFTransformer.git
$ cd MOFTransformer
$ pip install -e .
```

## 3. Download model and data
You can download various data and models used in `MOFTransformer` through the command line or python code.

1) (requirement) Pre-trained models (ckpt files of `PMTransformer`, `MOFTransformer`)
2) Fine-tuned models (h2 uptake and band gap)
3) The pre-embeddings for CoREMOF database
4) The pre-embeddings for QMOF database


### Download using command-line
You can download the file through the following command.
```bash
$ moftransformer download [target] (--outdir outdir) (--remove_tarfile)
```
Each argument is as follows:
- target : One or more of the `pretrain_model`, `finetuned_model`, `coremof`, `qmof`
- outdir (--outdir, -o) : (optional) Directory to save model or dataset.
  - default `pretrain_model` : [moftransformer_dir]/database/pmtransformer.ckpt or moftransformer.ckpt
  - default `finetuned_model` : [moftransformer_dir]/database/finetuend_model/
  - default `coremof` : [moftransformer_dir]/database/coremof/
  - default `qmof` : [moftransformer_dir]/database/qmof/


```bash
# download pre-trained model (required)
$ moftransformer download pretrain_model

# download graph-data and graph-data for CoREMOF (optional)
$ moftransformer download coremof

# download graph-data and graph-data for QMOF (optional)
$ moftransformer download qmof

# download fine-tuned model (optional)
$ mofransformer download finetuned_model
```

### Download using python
Another method is to use the python code.\
Commonly, it has two **optional** factors `direc` and `remove_tarfile`, which are the same as above.

```python
from moftransformer.utils.download import (
    download_pretrain_model,
    download_qmof,
    download_coremof,
    download_hmof,
    download_finetuned_model
)

# download pre-trained model
download_pretrain_model()
# download coremof
download_coremof()
# download qmof
download_qmof()
# download finetuned_model
download_finetuned_model()
```


## 4. Install GRIDAY (Optional)

If you want to calculate energy grids with cif files, you can use GRIDAY.
A `GRIDAY` is a tool for calculating energy grids shape of porous materials. (reference : https://github.com/Sangwon91/GRIDAY)

![GRIDAY](https://raw.githubusercontent.com/Sangwon91/GRIDAY/master/doc/img.png)

### 

### Installation using command-line

The simplest way is to use console scripts in bash.

```bash
$ moftransformer install-griday
```



### Installation using python

Alternatively, it can be installed by running the following function on Python.

```python
from moftransformer.utils import install_griday

install_griday()
```



### Installation using make

If the installation is not done perfectly, you can go directly to the path and install it.

The c++14 version is required to use the `GRIDAY`. In anaconda virtual environment, the corresponding version can be installed as follows when c++ version is incorrect.

```bash
$ conda install -c conda-forge gcc=9.5.0
$ conda install -c conda-forge gxx=9.5.0
```

Once the correct installation of g++ is completed, the `GRIDAY` could be installed in the following way.

```bash
$ cd [PATH_MOFTransformer]/libs/GRIDAY/  # move to path of griday-file
$ make              # make Makefile
$ cd scripts/
$ make              # make Makefile
```

If the `grid_gen` file is created in `scripts/`, it is installed.
