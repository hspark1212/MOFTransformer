# Training
Try to run `run.py` with hyperparameters in `model/config.py`
```angular2html
=== sacred === (https://sacred.readthedocs.io/en/stable/index.html)
Every experiment is sacred
Every experiment is great
If an experiment is wasted
God gets quite irate
```
## 1. Pretraining
If you train MTP (MOF Topology Prediction), run as bellows.
```angular2html
python run.py with task_mtp
```

## 2. Finetuning
if you predict bandgap properties with the pretrained model, load the pretrained model and run as bellows.
```angular2html
python run.py with downstream_bandgap load_path=best_ckpt/best_mtp_moc_vfp.ckpt
```

After running, the trained model, log, hyperparameter will be saved at `logdir` (in config.py).
Then you look over the results with `tensorboard`
```angular2html
tensorboard --logdir=result --bind_all
```

## 3. test
if you want to calculate scores with test dataset, 
you just set `test_only=True` and `load_path` and run like belows. 
```angular2html
python run.py with downstream_bandgap load_path=[--].cpkt test_only=True
```
