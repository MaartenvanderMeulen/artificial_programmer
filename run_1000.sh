for start_seed in `seq 1024 32 2047`
do
    echo start seed $start_seed
    end_seed = `expr start_seed + 31`
    for seed in `seq $start_seed 1 $end_seed`
    do
        if python solve_problems.py $seed a ; then killall python ; fi &
    done
    wait
    echo solved problem `grep solv tmp/a/lo* | wc -l` times
done