for start_seed in `seq 1024 32 2047`
do
    export end_seed=`expr $start_seed + 31`
    echo start seed $start_seed end seed $end_seed
    for seed in `seq $start_seed 1 $end_seed`
    do
        if python solve_problems.py $seed a 2>>err.txt ; then killall python ; fi &
    done
    wait
    echo solved problem `grep solv tmp/a/lo* | wc -l` times
done
