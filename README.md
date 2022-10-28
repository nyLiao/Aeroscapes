# Aeroscapes data
## Data
[Aeroscapes](https://github.com/ishann/aeroscapes)

Set the data path to args.data in train.py.


## Training 
```
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=3 --nnodes=1 train.py --num_epochs 30 --batch 2 --loss focalloss
```