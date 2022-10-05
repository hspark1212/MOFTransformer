# MOFTransformer

 Do you train machine learning models for every application? This package provides universal transfer learing for metal-organic frameworks(MOFs) to construct structure-property relationships. `MOFTransformer` obtains state-of-the-art performance to predict accross various properties that include gas adsorption, diffusion, electronic properties regardless of gas types. Beyond its universal transfer learning capabilityies, it provides feature importance analysis from its attentions scores to capture chemical intution.
 
![fig1](https://user-images.githubusercontent.com/64190846/167797065-1a104b35-a949-4775-93d4-c7310d90afbb.jpg)

## Architectures
`MOFTransformer`is a multi-modal Transformer pre-trained with 1 million hypothetical MOFs so that it efficiently capture both local and global feeatures of MOFs.

- `MOFformer` takes two different representations as input
  - Atom-based Graph Embedding : CGCNN w/o pooling layer -> local features
  - Energy-grid Embedding : 1D flatten patches of 3D energy grid -> global features
![fig2](https://user-images.githubusercontent.com/64190846/167792454-32ea32ad-29ba-4230-a15d-7e51c3ce8412.jpg)

## Install

## Getting Started

## Feature Importance Anaylsis
you can easily visualize feature importance analysis of atom-based graph embeddings and energy-grid embeddings.
```
%matplotlib widget
from visualize import PatchVisualizer

model_path = "examples/finetuned_bandgap.ckpt" # or 'examples/finetuned_h2_uptake.ckpt'
data_path = 'examples/visualize/dataset/'
cifname = 'MIBQAR01_FSR'

vis = PatchVisualizer.from_cifname(cifname, model_path, data_path)
vis.draw_graph()
```
![vis1](docs/source/getting_started/assets/1.gif)

```
vis = PatchVisualizer.from_cifname(cifname, model_path, data_path)
vis.draw_grid()
```
![vis2](https://github.com/hspark1212/MOFTransformer/blob/modify_readme/docs/source/getting_started/assets/1.gif =100x)
<img src="https://github.com/hspark1212/MOFTransformer/blob/modify_readme/docs/source/getting_started/assets/1.gif" width="100">
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
### reference
1. [Prediction of O2/N2 Selectivity in Metal−Organic Frameworks via High-Throughput Computational Screening and Machine Learning](https://pubs.acs.org/doi/abs/10.1021/acsami.1c18521)
2. [Using Machine Learning and Data Mining to Leverage Community Knowledge for the Engineering of Stable Metal–Organic Frameworks](https://pubs.acs.org/doi/10.1021/jacs.1c07217)
3. [Understanding the diversity of the metal-organic framework ecosystem](https://www.nature.com/articles/s41467-020-17755-8)
