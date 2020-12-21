#!/bin/bash
id=$1
mkdir -p tmp/$id
tsp -S 16
for seed in `seq 1000 1 1990`
do
    echo tsp $seed
    tsp -n -L $seed python solve_problems.py $seed experimenten/params_$id.txt
done
for t in `seq 17 1 31`
do
    echo sleep 60
    sleep 60
    echo tsp -S $t
    tsp -S $t
done
