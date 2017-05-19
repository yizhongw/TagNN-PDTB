#!/usr/bin/env bash
#echo 1
#python3 main.py --train --encoder binary-tree-lstm --attention --lr 0.1 --epochs 6 # very bad
#echo 2
#python3 main.py --train --encoder binary-tree-lstm --attention --lr 0.05 --epochs 6
#echo 3
#python3 main.py --train --encoder binary-tree-lstm --attention --lr 0.001 --epochs 10
echo 4
python3 main.py --train --encoder binary-tree-lstm --attention --wd 0.001 --epochs 6 # very bad
echo 5
python3 main.py --train --encoder binary-tree-lstm --attention --wd 0.00001 --epochs 6 # very bad
echo 6
python3 main.py --train --encoder binary-tree-lstm --attention --wd 0.000001 --epochs 6
echo 7
python3 main.py --train --encoder binary-tree-lstm --attention --batch_size 10 --epochs 6
echo 8
python3 main.py --train --encoder binary-tree-lstm --attention --batch_size 32 --epochs 10
echo 9
python3 main.py --train --encoder binary-tree-lstm --attention --batch_size 50 --epochs 15

