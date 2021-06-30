python -m torch.distributed.launch --nproc_per_node=8 fine-tune.py > logger/log.out 2>&1 
