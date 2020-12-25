#!/bin/bash
mkdir -p tmp/$1
for seed in `seq 1000 1 1030`
do
    python solve_problems.py $seed experimenten/params_$1.txt &
done
