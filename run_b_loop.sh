for iter in `seq 1 1 10`
do
    echo launch iter $iter
    for seed in `seq 1024 1 1151`
    do
        tsp -n python solve_problems.py $seed bL
    done
    tsp -w
    sleep 10
    mv tmp/bL tmp/bL$iter
    rm -r tmp/bL_input
    cp -r tmp/bL$iter tmp/bL_input
    mkdir bL
done
