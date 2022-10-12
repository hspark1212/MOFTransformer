# Installation

## 1. Requirements

- python >= 3.8 or newer
- Numpy



## 2. Installation

### Installation using PIP

The simplest way for installing `moftransformer` is to use PIP.

```bash
$ pip install moftransformer
```

If you want to deal with docs, use:

```bash
$ pip install moftransformer[docs]
```



### Installation from git clone

Or you can download it directly from github and install it.

```bash
$ git clone https://github.com/hspark1212/MOFTransformer.git
$ cd moftransformer
$ python setup.py install
```


### Editable install
If you want to modify the code, you can use the develop mode.

```bash
$ git clone https://github.com/hspark1212/MOFTransformer.git
$ cd moftransformer
$ pip install -e
```


## 3. Download model and data
You can download various data needed for MOFTransformer. \
(URL : https://figshare.com/articles/dataset/MOFTransformer/21155506)\
There are five data that you can download
1) Pre-trained model (MTP & MOC &VFP)
2) Fine-tuned model (h2 uptake and band gap)
3) Database which contain graph-data and grid-data for CoREMOF (~20,000)
4) Database which contain graph-data and grid-data for QMOF (~20,000)
5) Database which contain graph-data and grid-data for hMOF (~1M)


### Download using command-line
You can download the file through the following command.
```bash
$ moftransformer download [target] (--outdir outdir) (--remove_tarfile)
```
Each argument is as follows:
- target : One or more of the `pretrain_model`, `finetuned_model`, `coremof`, `qmof`, and `hmof`
- outdir (--outdir, -o) : (optional) Directory to save model or dataset.
  - default `pretrain_model` : [moftransformer_dir]/database/pretrained_model.ckpt
  - default `finetuned_model` : [moftransformer_dir]/database/finetuend_model/
  - default `coremof` : [moftransformer_dir]/database/coremof/
  - default `qmof` : [moftransformer_dir]/database/qmof/
  - default `hmof` : [moftransformer_dir]/database/hmof/
- remove_tarfile (--remove_tarfile, -r) : (optional) If activate, remove the downloaded tar.gz file


```bash
# download pre-trained model
$ moftransformer download pretrain_model

# download graph-data and graph-data for CoREMOF (optional)
$ moftransformer download coremof

# download graph-data and graph-data for QMOF (optional)
$ moftransformer download qmof

# download graph-data and graph-data for hMOF (optional)
$ moftransformer download hmof

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
# download hmof
download_hmof()
# download finetuned_model
download_finetuned_model()
```


## 4. Install GRIDAY

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

The c++14 version is required to use the `GRIDAY`. In anaconda virtual environment, the corresponding version can be installed as follows.

```bash
$ conda install -c conda-forge gcc=11.2.0
$ conda install -c conda-forge gxx=11.2.0
$ conda install -c anaconda make
```

Once the correct installation of g++ is completed, the `GRIDAY` could be installed in the following way.

```bash
$ cd [PATH_MOFTransformer]/libs/GRIDAY/  # move to path of griday-file
$ make              # make Makefile
$ cd scripts/
$ make              # make Makefile
```

If the `grid_gen` file is created in `scripts/`, it is installed.
