![](https://raw.githubusercontent.com/hspark1212/MOFTransformer/master/docs/source/assets/fig1.jpg)

<p align="center">
 <a href="https://hspark1212.github.io/MOFTransformer/">
     <img alt="Docs" src="https://img.shields.io/badge/Docs-v1.0.2-brightgreen.svg?style=plastic">
 </a>
  <a href="https://pypi.org/project/moftransformer/">
     <img alt="PypI" src="https://img.shields.io/badge/PyPI-v1.0.2-blue.svg?style=plastic&logo=PyPI">
 </a>
  <a href="https://doi.org/10.6084/m9.figshare.21155506.v2">
     <img alt="Figshare" src="https://img.shields.io/badge/Figshare-v2-blue.svg?style=plastic&logo=figshare">
 </a>
 <a href="https://chemrxiv.org/engage/chemrxiv/article-details/634fbf8a4a18764f58e9fda5">
     <img alt="DOI" src="https://img.shields.io/badge/DOI-doi-organge.svg?style=plastic">
 </a>
 <a href="https://pypi.org/project/moftransformer/">
     <img alt="Lincense" src="https://img.shields.io/badge/License-MIT-lightgrey.svg?style=plastic">
 </a>
</p>

# [MOFTransformer](https://hspark1212.github.io/MOFTransformer/index.html)

 This package provides universal transfer learing for metal-organic frameworks(MOFs) to construct structure-property relationships. `MOFTransformer` obtains state-of-the-art performance to predict accross various properties that include gas adsorption, diffusion, electronic properties regardless of gas types. Beyond its universal transfer learning capabilityies, it provides feature importance analysis from its attentions scores to capture chemical intution.

## [Install](https://hspark1212.github.io/MOFTransformer/installation.html)

### Depedencies
```
python>=3.8
```
Given that MOFTransformer is based on pytorch, please install pytorch (>= 1.10.0) according to your environments.

### Installation using PIP 
```
$ pip install moftransformer
```

### Download the pretrained model (ckpt file)
- you can download the pretrained model with 1 M hMOFs in [figshare](https://figshare.com/articles/dataset/MOFTransformer/21155506)
or you can download with a command line:
```
$ moftransformer download pretrain_model
```
### (Optional) Download dataset for CoREMOF, QMOF
- we've provide the dataset of MOFTransformer (i.e., atom-based graph embeddings and energy-grid embeddings) for CoREMOF, QMOF
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
`MOFTransformer`is a multi-modal Transformer pre-trained with 1 million hypothetical MOFs so that it efficiently capture both local and global feeatures of MOFs.

- `MOFformer` takes two different representations as input
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
| Property                                 | MOFTransformer | Original Paper | Number of Data | Remarks          | Reference |
|------------------------------------------|----------------|----------------|----------------|------------------|-----------|
|N<sub>2</sub> uptake                     | R2: 0.78       | R2: 0.71       | 5,286          | CoRE MOF         | 1         |
|O<sub>2</sub> uptake                     | R2: 0.83       | R2: 0.74       | 5,286          | CoRE MOF         | 1         |
|N<sub>2</sub> diffusivity                | R2: 0.77       | R2: 0.76       | 5,286          | CoRE MOF         | 1         |
|O<sub>2</sub> diffusivity                | R2: 0.78       | R2: 0.74       | 5,286          | CoRE MOF         | 1         |
|CO<sub>2</sub> Henry coefficient         | MAE : 0.30     | MAE : 0.42     | 8,183          | CoRE MOF         | 2         |
|Solvent removal stability classification | ACC : 0.76     | ACC : 0.76     | 2,148          | Text-mining data | 3         |
|Thermal stability regression             | R2 : 0.44      | R2 : 0.46      | 3,098          | Text-mining data | 3         |
### Reference
1. [Prediction of O2/N2 Selectivity in Metal−Organic Frameworks via High-Throughput Computational Screening and Machine Learning](https://pubs.acs.org/doi/abs/10.1021/acsami.1c18521)
2. [Using Machine Learning and Data Mining to Leverage Community Knowledge for the Engineering of Stable Metal–Organic Frameworks](https://pubs.acs.org/doi/10.1021/jacs.1c07217)
3. [Understanding the diversity of the metal-organic framework ecosystem](https://www.nature.com/articles/s41467-020-17755-8)
