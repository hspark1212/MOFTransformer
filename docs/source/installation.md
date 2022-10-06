# Installation

## 1. Requirements

- python >= 3.8 or newer
- Numpy



## 2. Installation

### Installation using PIP (not yet)

The simplest way is install `moftransformer` is to use PIP.

```bash
$ pip install moftransformer
```

If you want to deal with docs, use:

```bash
$ pip install moftransformer[docs]
```



### Installation from git clone

Or you can download it directly from github and install it.

```
$ git clone https://github.com/hspark1212/MOFTransformer.git
$ cd moftransformer
$ python setup.py install
```



If you want to modify the code, you can use the develop mode.

```develop
$ python setup.py develop
```



## 3. Install GRIDAY

A `GRIDAY` is a tool for calculating energy grids shape of porous materials. (reference : https://github.com/Sangwon91/GRIDAY)

![GRIDAY](https://raw.githubusercontent.com/Sangwon91/GRIDAY/master/doc/img.png)

### 

### Installation using bash

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
conda install -c conda-forge gcc=11.2.0
conda install -c conda-forge gxx=11.2.0
conda install -c anaconda make
```

Once the correct installation of g++ is completed, the `GRIDAY` could be installed in the following way.

```bash
$ cd [PATH_MOFTransformer]/libs/GRIDAY/  # move to path of griday-file
$ make              # make Makefile
$ cd scripts/
$ make              # make Makefile
```

If the `grid_gen` file is created in `scripts/`, it is installed.
