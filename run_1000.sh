#!/bin/bash
id=$1
mkdir -p tmp/$id
for seed in `seq 1030 1 1999`
do
    tsp -n -L $seed python solve_problems.py $seed experimenten/params_$id.txt
done
