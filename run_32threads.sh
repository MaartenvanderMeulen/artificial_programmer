#!/bin/bash
for id in $*
do
    mkdir -p tmp/$id
    nt=28
    lastt=`expr $nt - 1`
    for t in `seq 0 1 $lastt`
    do
        startseed=`expr 1000 + $t`
        for seed in `seq $startseed $nt 1999` ; do python solve_problems.py $seed experimenten/params_$id.txt ; done &
    done
    wait
    echo $id `date` `grep solved tmp/$id/log*.txt | wc --lines`
done
