#!/bin/bash
id=$1
mkdir -p tmp/$id
tsp -S 24
for seed in `seq 1000 1 1990`
do
    echo tsp $seed
    tsp -n -L $seed python solve_problems.py $seed experimenten/params_$id.txt
done
