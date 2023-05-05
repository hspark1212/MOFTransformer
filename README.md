![](https://raw.githubusercontent.com/hspark1212/MOFTransformer/master/docs/source/assets/fig1.jpg)

<p align="center">
 <a href="https://hspark1212.github.io/MOFTransformer/">
     <img alt="Docs" src="https://img.shields.io/badge/Docs-v2.0.1-brightgreen.svg?style=plastic">
 </a>
  <a href="https://pypi.org/project/moftransformer/">
     <img alt="PypI" src="https://img.shields.io/badge/PyPI-v2.0.1-blue.svg?style=plastic&logo=PyPI">
 </a>
  <a href="https://doi.org/10.6084/m9.figshare.21155506.v2">
     <img alt="Figshare" src="https://img.shields.io/badge/Figshare-v2-blue.svg?style=plastic&logo=figshare">
 </a>
 <a href="https://doi.org/10.5281/zenodo.7593333">
     <img alt="DOI" src="https://img.shields.io/badge/DOI-doi-organge.svg?style=plastic">
 </a>
 <a href="https://pypi.org/project/moftransformer/">
     <img alt="Lincense" src="https://img.shields.io/badge/License-MIT-lightgrey.svg?style=plastic">
 </a>
</p>

# [PMTransformer (MOFTransformer)](https://hspark1212.github.io/MOFTransformer/index.html)

 This package provides a universal transfer learning model, `PMTransformer` (Porous Materials Transformer), which obtains the state-of-the-art performance in predicting various properties of porous materials. The PMTRansformer was pre-trainied with 1.9 million hypothetical porous materials including Metal-Organic Frameworks (MOFs), Covalent-Organic Frameworks (COFs), Porous Polymer Networks (PPNs), and zeolites. By fine-tuning the pre-trained `PMTransformer`, you can easily obtain machine learning models to accurately predict various properties of porous materials .
 
 NOTE: From version 2.0.0, the default pre-training model has been changed from `MOFTransformer` to `PMTransformer`, which was pre-trained with a larger dataset, containing other porous materials as well as MOFs. The `PMTransformer` outperforms the `MOFTransformer` in predicting various properties of porous materials.

## [Install](https://hspark1212.github.io/MOFTransformer/installation.html)

### Depedencies
```
python>=3.8
```
Given that MOFTransformer is based on pytorch, please install pytorch (>= 1.12.0) according to your environments.

### Installation using PIP 
```
$ pip install moftransformer
```

### Download the pretrained models (ckpt file)
- you can download the pretrained models (`PMTransformer.ckpt` and `MOFTransformer.ckpt`) [figshare](https://figshare.com/articles/dataset/PMTransformer_pre-trained_model/22698655/2)

or you can download with a command line:
```
$ moftransformer download pretrain_model
```
### (Optional) Download pre-embeddings for CoREMOF, QMOF
- we've provide the pre-embeddings (i.e., atom-based graph embeddings and energy-grid embeddings), inputs of `PMTransformer`, for CoREMOF, QMOF database.
```
$ moftransformer download coremof
$ moftransformer download qmof
```

## [Getting Started](https://hspark1212.github.io/MOFTransformer/tutorial.html)
1. At first, you download dataset of hMOFs (20,000 MOFs) as an example.
```
$ moftransformer download hmof
```
2. Fine-tune the pretrained MOFTransformer.
```python
import moftransformer
from moftransformer.examples import example_path

# data root and downstream from example
data_root = example_path['data_root']
downstream = example_path['downstream']
log_dir = './logs/'
# load_path = "pmtransformer" (default)

moftransformer.run(data_root, downstream, log_dir=log_dir, 
                   max_epochs=max_epochs, batch_size=batch_size,)
```
3. Visualize analysis of feature importance for the fine-tuned model.
```python
%matplotlib widget
from visualize import PatchVisualizer

model_path = "examples/finetuned_bandgap.ckpt" # or 'examples/finetuned_h2_uptake.ckpt'
data_path = 'examples/visualize/dataset/'
cifname = 'MIBQAR01_FSR'

vis = PatchVisualizer.from_cifname(cifname, model_path, data_path)
vis.draw_graph() # or vis.draw_grid()
```

## [Architecture](https://hspark1212.github.io/MOFTransformer/introduction.html)
It is a multi-modal pre-training Transformer encoder which is designed to capture both local and global features of porous materials. 

The pre-traning tasks are as follows:
(1) Topology Prediction
(2) Void Fraction Prediction
(3) Building Block Classification
 
It takes two different representations as input
  - Atom-based Graph Embedding : CGCNN w/o pooling layer -> local features
  - Energy-grid Embedding : 1D flatten patches of 3D energy grid -> global features
  
<p align="center">
  <img src="https://raw.githubusercontent.com/hspark1212/MOFTransformer/master/docs/source/assets/fig2.jpg" width="700")
</p>

## [Feature Importance Anaylsis](https://hspark1212.github.io/MOFTransformer/getting_started/visualization.html)
you can easily visualize feature importance analysis of atom-based graph embeddings and energy-grid embeddings.
```python
%matplotlib widget
from visualize import PatchVisualizer

model_path = "examples/finetuned_bandgap.ckpt" # or 'examples/finetuned_h2_uptake.ckpt'
data_path = 'examples/visualize/dataset/'
cifname = 'MIBQAR01_FSR'

vis = PatchVisualizer.from_cifname(cifname, model_path, data_path)
vis.draw_graph()
```
<p align="center">
<img src="https://raw.githubusercontent.com/hspark1212/MOFTransformer/master/docs/source/getting_started/assets/1.gif" width="400">
</p>

```python
vis = PatchVisualizer.from_cifname(cifname, model_path, data_path)
vis.draw_grid()
```
<p align="center">
<img src="https://raw.githubusercontent.com/hspark1212/MOFTransformer/master/docs/source/getting_started/assets/3.gif" width="400">
</p>

## Universal Transfer Learning

Comparison of mean absolute error (MAE) values for various baseline models, scratch, MOFTransformer, and PMTransformer on different properties of MOFs, COFs, PPNs, and zeolites. The bold values indicate the lowest MAE value for each property. The details of information can be found in [PMTransformer paper]()

| Material | Property | Number of Dataset | Energy histogram | Descriptor-based ML | CGCNN | Scratch | MOFTransformer | PMTransformer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MOF | H<sub>2</sub> Uptake (100 bar) | 20,000 | 9.183 | 9.456 | 32.864 | 7.018 | 6.377 | **5.963** |
| MOF | H<sub>2</sub> diffusivity (dilute) | 20,000 | 0.644 | 0.398 | 0.6600 | 0.391 | 0.367 | 0.**366** |
| MOF | Band-gap | 20.373 | 0.913 | 0.590 | 0.290 | 0.271 | 0.224 | **0.216** |
| MOF | N<sub>2</sub> uptake (1 bar) | 5,286 | 0.178 | 0.115 | 0.108 | 0.102 | 0.071 | **0.069** |
| MOF | O<sub>2</sub> uptake (1 bar) | 5,286 | 0.162 | 0.076 | 0.083 | 0.071 | **0.051** | 0.053 |
| MOF | N<sub>2</sub> diffusivity (1 bar) | 5,286 | 7.82e-5 | 5.22e-5 | 7.19e-5 | 5.82e-05 | **4.52e-05** | 4.53e-05 |
| MOF | O<sub>2</sub> diffusivity (1 bar) | 5,286 | 7.14e-5 | 4.59e-5 | 6.56e-5 | 5.00e-05 | 4.04e-05 | **3.99e-05** |
| MOF | CO<sub>2</sub> Henry coefficient | 8,183 | 0.737 | 0.468 | 0.426 | 0.362 | 0.295 | **0.288** |
| MOF | Thermal stability | 3,098 | 68.74 | 49.27 | 52.38 | 52.557 | 45.875 | **45.766** |
| COF | CH<sub>4</sub> uptake (65bar) | 39,304 | 5.588 | 4.630 | 15.31 | 2.883 | 2.268 | **2.126** |
| COF | CH<sub>4</sub> uptake (5.8bar) | 39,304 | 3.444 | 1.853 | 5.620 | 1.255 | **0.999** | 1.009 | 
| COF | CO<sub>2</sub> heat of adsorption | 39,304 | 2.101 | 1.341 | 1.846 | 1.058 | 0.874 | **0.842** |
| COF | CO<sub>2</sub> log KH | 39,304 | 0.242 | 0.169 | 0.238 | 0.134 | 0.108 | **0.103** |
| PPN | CH<sub>4</sub> uptake (65bar) | 17,870 | 6.260 | 4.233 | 9.731 | 3.748 | 3.187 | **2.995** | 
| PPN | CH<sub>4</sub> uptake (1bar) | 17,870  | 1.356	| 0.563	| 1.525	| 0.602	| 0.493	| **0.461** | 
| Zeolite | CH<sub>4</sub>  KH (unitless) | 99,204	| 8.032	| 6.268	| 6.334	| 4.286	| 4.103	| **3.998** |
| Zeolite | CH<sub>4</sub>  Heat of adsorption | 99,204	| 1.612	|1.033	| 1.603	| 0.670	| 0.647	|**0.639** |

