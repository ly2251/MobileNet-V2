

* If you want to train from scratch, you can run as follows:

```
python train.py --batch-size 256 --gpus 0,1,2,3
```

* If you want to train from one checkpoint, you can run as follows(for example train from `epoch_4.pth.tar`, the `--start-epoch` parameter is corresponding to the epoch of the checkpoint):

```
python train.py --batch-size 256 --gpus 0,1,2,3 --resume output/epoch_4.pth.tar --start-epoch 4
```