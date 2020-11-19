#!/usr/bin/bash
for id in $*
do
    rm -rf tmp/$id
    mkdir -p tmp/$id
    for part in A B C D
    do
        ./experimenten/core_00_$part.sh $id &
        ./experimenten/core_01_$part.sh $id &
        wait
        ./experimenten/core_02_$part.sh $id &
        ./experimenten/core_03_$part.sh $id &
        wait
        echo $id part $part `date` `grep solved tmp/$id/log*.txt | wc --lines` >> tmp/$id/resultaat_$part.$id.txt
        cat tmp/$id/resultaat_$part.$id.txt        
    done
    cat tmp/$id/resultaat_D.$id.txt >> experimenten/resultaat_$id.txt
done
