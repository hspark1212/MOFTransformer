MOFTransformer
===============

.. note::
   From version 2.0.0, the default pre-training model has been changed from MOFTransformer to PMTransformer. 

MOFTransformer (or PMTransformer) is a Python library that focuses on structure-property relationships in porous materials including Metal-Organic Frameworks (MOFs), Covalent-Organic Frameworks (COFs), Porous Polymer Networks (PPNs), and zeolites 
The multi-modal pre-trianing Transformer showcases remarkable transfer learning capabilities across various properties of porous materials.
With MOFTrasformer, there is no need to develop and train machine learning models to predict different properties for different applications from scratch. 
The library provides tools for fine-tuning, pre-training, and feature importance analysis using attention scores.


Features
--------
- The library provides a pre-trained PMTransformer `ckpt file <https://figshare.com/articles/dataset/PMTransformer_pre-trained_model/22698655/2>`_ with 1.9 million hypothetical porous materials. we provide  of PMTransformer pre-trained with 1.9 million hypothetical porous materials.
- With fine-tuning, the pre-training model allows for high-performance machine learning models to predict properties of porous materials.
- `The pre-embeddings <https://figshare.com/articles/dataset/MOFTransformer/21155506>`_ (i.e., atom-based embeddings and energy-grid embeddings) for CoRE MOF, QMOF databases are available.
- Feature importance analysis can be easily visualized from attention scores of the fine-tuning models.


**atom-base graph embedding**

.. image:: getting_started/assets/1.gif
   :width: 800

**energy-grid embedding**

.. image:: getting_started/assets/6.gif
   :width: 800

**patches of energy-grid embedding**

.. image:: getting_started/assets/7.gif
   :width: 800

.. image:: getting_started/assets/8.gif
   :width: 800

Contents
--------
.. toctree::
   :titlesonly:
   :maxdepth: 2

   introduction
   installation
   dataset
   getting_started
   tutorial

Indices and Tables
------------------
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`








