for id in 02
do
    mkdir -p tmp/$id
    ./experimenten/core_00.sh $id &
    ./experimenten/core_01.sh $id &
    ./experimenten/core_02.sh $id &
    ./experimenten/core_03.sh $id &
    echo waiting $id ...
    wait
    echo done $id
    grep solved tmp/$id/log*.txt | wc --lines > tmp/$id/resultaat_$id.txt
done
