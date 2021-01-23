for x in a b
do
    rm tmp/$x/*
    for seed in `seq 1024 1 2047`
    do
        tsp -n python solve_problems.py $seed $x
    done
done
