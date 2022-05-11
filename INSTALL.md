#  Installation

## 1. Install pretrain_mof
- 나중에 추가
- requirements.txt
- setup.py


## 2. install griday
A **griday** is a tool for obtaining an energy grid of a material.

The c++14 version is required to use the griday.

In anaconda virtual environment, the corresponding version can be installed as follows.

```bash
$ conda install -c conda-forge gcc=11.2.0
$ conda install -c conda-forge gxx=11.2.0
$ conda install -c anaconda make
```

Once the correct installation of g++ is completed, the grid may be installed in the following way.

```bash
$ cd [griday-path]  # move to path of griday-file
$ make              # make Makefile
$ cd script
$ make              # make Makefile

```
