#!/bin/bash 

# Author: Wei-Ning Hsu

vqon=$1    # 01000
name=$2    # RDVQ_00000_01000
args=$3    # "--resume True --seed_dir <dir>"
data_tr="/home/harwath/data/SpokenCOCO/SpokenCOCO_train.json"
data_dt="/home/harwath/data/SpokenCOCO/SpokenCOCO_val.json"
data_mt="/home/harwath/data/SpokenCOCO/train_ctm.txt"
export data_mt
vqonarg=$(echo $vqon | sed 's/./&,/g' | sed 's/,$//g')  # insert ',' in between
rm -rf exps/

echo $vqon
echo $vqonarg
vqonarg='0,0,0,0,0'
train_decay=$(echo "scale=3;985/1000" | bc)
rm -rf logs/

train_rate=$(echo "scale=5; 2/10000" | bc)
dropout=$(echo "scale=2;45/100" | bc)
expdir="./exps/$name"

python3 ../run_ResDavenetVQ.py --mode eval \
    --VQ-turnon $vqonarg --exp-dir $expdir \
    --data-train $data_tr --data-val $data_dt \
    --batch-size 128 --n-epochs 20 --save-every -1\
    --num-words-dropout 2 --word-dropout $dropout\
    --lr $train_rate --lr-decay-multiplier $train_decay --lr-decay 3\
    $args > "logs/train.out" 2>&1 &

