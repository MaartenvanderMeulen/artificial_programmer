#!/usr/bin/bash
for id in $*
do
    rm -rf tmp/$id
    mkdir -p tmp/$id
    /c/Users/Maarten/Miniconda3/python search.py 100 experimenten/params_$id.txt &
    /c/Users/Maarten/Miniconda3/python search.py 101 experimenten/params_$id.txt &
    wait
    echo $id `date` `grep solved tmp/$id/log*.txt | wc --lines`
done
