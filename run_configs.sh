for nr in 01
do
    cp tmp_run/tmp_config_$nr.txt config.txt
    ./tmp_run/tmp_core_00.sh &
    ./tmp_run/tmp_core_01.sh &
    ./tmp_run/tmp_core_02.sh &
    ./tmp_run/tmp_core_03.sh &
    echo waiting $nr ...
    wait
    echo done $nr
    grep solved tmp/*.txt | wc --lines > tmp_run/tmp_solved_$nr.txt
done
