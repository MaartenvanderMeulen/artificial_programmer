#!/bin/bash
id=$1
mkdir -p tmp/$id
nt=30
lastt=`expr $nt - 1`
for t in `seq 0 1 $lastt`
do
    startseed=`expr 1000 + $t`
    for seed in `seq $startseed $nt 1999` ; do python solve_problems.py $seed experimenten/params_$id.txt ; done &
done
