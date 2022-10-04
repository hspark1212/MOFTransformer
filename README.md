# MOFTransformer

 Do you train machine learning models for every application? This package provides universal transfer learing for metal-organic frameworks(MOFs). `MOFTransformer` is a multi-modal pretraining Transformer that facilaites capturing both local and global feeatures of MOFs. It obtains state-of-the-art performance to predict accross various properties that include gas adsorption, diffusion, electronic properties regardless of gas types. Beyond its universal transfer learning capabilityies, it provides feature importance analysis from its attentions scores to capture chemical intution.
![fig1](https://user-images.githubusercontent.com/64190846/167797065-1a104b35-a949-4775-93d4-c7310d90afbb.jpg)

## Architectures
![fig2](https://user-images.githubusercontent.com/64190846/167792454-32ea32ad-29ba-4230-a15d-7e51c3ce8412.jpg)
- `MOFformer` takes two different representations as input
1) Atom-based Graph Embedding : CGCNN w/o pooling layer -> local features
2) Energy-grid Embedding : 1D flatten patches of 3D energy grid -> global features

## Install

## Getting Started

## Visualize

## UTL




### 2. UTL (in progress)

![image](https://user-images.githubusercontent.com/64190846/171344412-c43cbf12-adc3-41ab-86ef-f4d65ea35765.png)

1. [Prediction of O2/N2 Selectivity in Metal−Organic Frameworks via High-Throughput Computational Screening and Machine Learning](https://pubs.acs.org/doi/abs/10.1021/acsami.1c18521)
2. [Using Machine Learning and Data Mining to Leverage Community Knowledge for the Engineering of Stable Metal–Organic Frameworks](https://pubs.acs.org/doi/10.1021/jacs.1c07217)
3. [Understanding the diversity of the metal-organic framework ecosystem](https://www.nature.com/articles/s41467-020-17755-8)
