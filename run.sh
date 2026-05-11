#!/bin/sh

model=MetaDist # MetaDist GraD Meta-Self Metacon_S Metacon_D
dataset=cora # cora citeseer polblogs cora_ml
ptb_rate=0.05
start_seed=10

python attack.py --model $model --dataset $dataset --ptb_rate $ptb_rate

python test.py --model $model --dataset $dataset --victim GCN --ptb_rate $ptb_rate --start_seed $start_seed
python test.py --model $model --dataset $dataset --victim GAT --ptb_rate $ptb_rate --start_seed $start_seed

# python test.py --model $model --dataset $dataset --victim RGCN --ptb_rate $ptb_rate --start_seed $start_seed
# python test.py --model $model --dataset $dataset --victim ProGNN --ptb_rate $ptb_rate --start_seed $start_seed
# python test.py --model $model --dataset $dataset --victim SimPGCN --ptb_rate $ptb_rate --start_seed $start_seed
