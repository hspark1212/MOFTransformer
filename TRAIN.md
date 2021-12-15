# Pretraining
## 1. train
Try to run `run.py` with hyperparameters in `model/config.py`
```angular2html
we totally agree with sacred 

=== sacred === (https://sacred.readthedocs.io/en/stable/index.html)
Every experiment is sacred
Every experiment is great
If an experiment is wasted
God gets quite irate
```
For example, training egcnn with 10k dataset is as bellows.
```angular2html
python run.py with egcnn_classification_10k
```
Then, the trined model, log, hyperparameter will be saved at `logdir` (in config.py).
Then you look over the results with `tensorboard`
```angular2html
tensorboard --logdir=result --bind_all
```

## 2. test
if you want to calculate scores with test dataset, 
you just set `test_only=True` and `load_path` and run like belows. 
```angular2html
python run.py with test_egcnn_classification_1k test_only=True load_path=result/[--].cpkt
```
