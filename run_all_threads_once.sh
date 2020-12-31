#!/bin/bash
id=$1
mkdir -p tmp/$id
for seed in `seq 1000 1 1029`
do
    tsp -n -L $seed python solve_problems.py $seed $id
done
