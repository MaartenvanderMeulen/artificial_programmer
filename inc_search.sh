#!/bin/bash
for id in $*
do
    for f in tmp/$id/log*.txt
    do
        if grep solved $f > /dev/null
        then
            :
        else
            rm $f
        fi
    done
    for seed in `seq 100 4 199` ; do python search.py $seed experimenten/params_$id.txt ; done &
    for seed in `seq 101 4 199` ; do python search.py $seed experimenten/params_$id.txt ; done &
    for seed in `seq 102 4 199` ; do python search.py $seed experimenten/params_$id.txt ; done &
    for seed in `seq 103 4 199` ; do python search.py $seed experimenten/params_$id.txt ; done &
    wait
    echo $id `date` `grep solved tmp/$id/log*.txt | wc --lines`
done
