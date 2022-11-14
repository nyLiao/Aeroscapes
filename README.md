# Aeroscapes data
## Data
[Aeroscapes](https://github.com/ishann/aeroscapes)

Set the data path to args.data in train.py.


## Commands
Training
```
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=3 --nnodes=1 train.py --num_epochs 30 --batch 2 --loss focalloss
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 train.py -c ./config/unet.json --nworker 12
```

Plot confusion matrix: `python test_confmat.py -c ./config/deeplabv3.json -f 1114`