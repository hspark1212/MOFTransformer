# Introduction

```
Note: From version 2.0.0, the default pre-training model has been changed from `MOFTransformer` to `PMTransformer`.
```

`PMTransformer` (Porous Materials Transformer), which obtains the state-of-the-art performance in predicting various properties of porous materials. The PMTRansformer was pre-trainied with 1.9 million hypothetical porous materials including Metal-Organic Frameworks (MOFs), Covalent-Organic Frameworks (COFs), Porous Polymer Networks (PPNs), and zeolites. By fine-tuning the pre-trained PMTransformer, you can easily obtain machine learning models to accurately predict various properties of porous materials.

`PMTransformer` was pre-trained with a larger dataset, containing other porous materials as well as MOFs. The `PMTransformer` outperforms the `MOFTransformer` in predicting various properties of porous materials.

## Pre-training
![fig2](https://user-images.githubusercontent.com/64190846/167792454-32ea32ad-29ba-4230-a15d-7e51c3ce8412.jpg)

It is a multi-modal pre-training Transformer encoder which is designed to capture both local and global features of porous materials. 

The pre-traning tasks are as follows:
(1) Topology Prediction
(2) Void Fraction Prediction
(3) Building Block Classification
 
It takes two different representations as input
  - Atom-based Graph Embedding : CGCNN w/o pooling layer -> local features
  - Energy-grid Embedding : 1D flatten patches of 3D energy grid -> global features
   
## Fine-tuning
- In the fine-tuning step, it is traind to predict the desired properties with the weights of the pre-trained model as initial weights.
- A single dense layer is added to [CLS] token for fine-tuning.

### Results
Gas uptake (H<sub>2</sub> uptake at 100 bar), Diffusivity (H<sub>2</sub> diffusivity), Electronic properties (PBE bandgap)

## Universal transfer learning
Comparison of mean absolute error (MAE) values for various baseline models, scratch, MOFTransformer, and PMTransformer on different properties of MOFs, COFs, PPNs, and zeolites. The bold values indicate the lowest MAE value for each property. The details of information can be found in [PMTransformer paper](https://chemrxiv.org/engage/chemrxiv/article-details/644a0651df78ec50157390c9)

| Material | Property | Energy histogram | Descriptor-based ML | CGCNN | Scratch | MOFTransformer | PMTransformer |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MOF | H<sub>2</sub> Uptake (100 bar) | 9.183 | 9.456 | 32.864 | 7.018 | 6.377 | **5.963** |
| MOF | H<sub>2</sub> diffusivity (dilute) | 0.644 | 0.398 | 0.6600 | 0.391 | 0.367 | 0.**366** |
| MOF | Band-gap | 0.913 | 0.590 | 0.290 | 0.271 | 0.224 | **0.216** |
| MOF | N<sub>2</sub> uptake (1 bar) | 0.178 | 0.115 | 0.108 | 0.102 | 0.071 | **0.069** |
| MOF | O<sub>2</sub> uptake (1 bar) | 0.162 | 0.076 | 0.083 | 0.071 | **0.051** | 0.053 |
| MOF | N<sub>2</sub> diffusivity (1 bar) | 7.82e-5 | 5.22e-5 | 7.19e-5 | 5.82e-05 | **4.52e-05** | 4.53e-05 |
| MOF | O<sub>2</sub> diffusivity (1 bar) | 7.14e-5 | 4.59e-5 | 6.56e-5 | 5.00e-05 | 4.04e-05 | **3.99e-05** |
| MOF | CO<sub>2</sub> Henry coefficient | 0.737 | 0.468 | 0.426 | 0.362 | 0.295 | **0.288** |
| MOF | Thermal stability | 68.74 | 49.27 | 52.38 | 52.557 | 45.875 | **45.766** |
| COF | CH<sub>4</sub> uptake (65bar) | 5.588 | 4.630 | 15.31 | 2.883 | 2.268 | **2.126** |
| COF | CH<sub>4</sub> uptake (5.8bar) | 3.444 | 1.853 | 5.620 | 1.255 | **0.999** | 1.009 | 
| COF | CO<sub>2</sub> heat of adsorption | 2.101 | 1.341 | 1.846 | 1.058 | 0.874 | **0.842** |
| COF | CO<sub>2</sub> log KH | 0.242 | 0.169 | 0.238 | 0.134 | 0.108 | **0.103** |
| PPN | CH<sub>4</sub> uptake (65bar) | 6.260 | 4.233 | 9.731 | 3.748 | 3.187 | **2.995** | 
| PPN | CH<sub>4</sub> uptake (1bar) | 1.356	| 0.563	| 1.525	| 0.602	| 0.493	| **0.461** | 
| Zeolite | CH<sub>4</sub>  KH (unitless) | 8.032	| 6.268	| 6.334	| 4.286	| 4.103	| **3.998** |
| Zeolite | CH<sub>4</sub>  Heat of adsorption | 1.612	|1.033	| 1.603	| 0.670	| 0.647	|**0.639** |

