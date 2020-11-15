#!/usr/bin/bash
for id in $*
do
    mkdir -p tmp/$id
    ./experimenten/core_00.sh $id &
    ./experimenten/core_01.sh $id &
    ./experimenten/core_02.sh $id &
    ./experimenten/core_03.sh $id &
    echo waiting $id ...
    wait
    grep solved tmp/$id/log*.txt | wc --lines > tmp/$id/resultaat_$id.txt
    echo score $id is `cat tmp/$id/resultaat_$id.txt`
done
