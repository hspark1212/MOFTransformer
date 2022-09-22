# Getting started

you need to download `best_mtp_moc_vfp.ckpt` that is the pretrained model with MTP + MOC + VFP tasks from [figshare](https://figshare.com/articles/dataset/MOFTransformer/21155506).

you can change parameters `model/config.py` for pre-training, fine-tuning, and test. 
## 1. Training
### Fine-tuning
 In order to load the wights of pre-trained model as initial weights, you should change `load_path` in `conf.py`.
The following script will fine-tune the pre-model with the `examples` directory
```python

```
### Pre-training

## 2. Test

## 3. Visualization