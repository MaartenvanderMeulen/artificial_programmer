#!/bin/bash
for id in $*
do
    for seed in `seq 1000 4 2999` ; do python search.py $seed experimenten/params_$id.txt ; done &
    for seed in `seq 1001 4 2999` ; do python search.py $seed experimenten/params_$id.txt ; done &
    for seed in `seq 1002 4 2999` ; do python search.py $seed experimenten/params_$id.txt ; done &
    for seed in `seq 1003 4 2999` ; do python search.py $seed experimenten/params_$id.txt ; done &
    wait
    echo $id `date` `grep solved tmp/$id/log*.txt | wc --lines`
done
