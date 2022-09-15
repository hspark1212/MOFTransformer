#  Installation

## 1. Install
`pip install -r requirements.txt`

## 2. Install griday
A [**griday**](https://github.com/Sangwon91/GRIDAY) is a tool for generating energy grids of a material.

The c++14 version is required to use the griday.

In anaconda virtual environment, the corresponding version can be installed as follows.

```bash
conda install -c conda-forge gcc=11.2.0
conda install -c conda-forge gxx=11.2.0
conda install -c anaconda make
```

Once the correct installation of g++ is completed, the gridday could be installed in the following way.

```bash
cd [griday-path]  # move to path of griday-file
make              # make Makefile
cd scripts
make              # make Makefile

```
