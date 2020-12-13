#!/bin/bash
mkdir -p tmp/$1
for t in `seq 0 1 23`
do
    python solve_problems.py `expr 1000 + $t` experimenten/params_$1.txt &
    sleep 50
done
