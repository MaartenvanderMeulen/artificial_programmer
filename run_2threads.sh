#!/usr/bin/bash
for id in $*
do
    rm -rf tmp/$id
    mkdir -p tmp/$id
    ./experimenten/core_00_A.sh $id &
    ./experimenten/core_01_A.sh $id &
    wait
    ./experimenten/core_02_A.sh $id &
    ./experimenten/core_03_A.sh $id &
    wait
    grep solved tmp/$id/log*.txt | wc --lines > tmp/$id/resultaat_$id.txt
    echo $id scores `cat tmp/$id/resultaat_$id.txt`
done
echo
for id in $*
do
    ./experimenten/core_00_B.sh $id &
    ./experimenten/core_01_B.sh $id &
    wait
    ./experimenten/core_02_B.sh $id &
    ./experimenten/core_03_B.sh $id &
    wait
    grep solved tmp/$id/log*.txt | wc --lines > tmp/$id/resultaat_$id.txt
    echo $id scores `cat tmp/$id/resultaat_$id.txt`
done
echo
for id in $*
do
    ./experimenten/core_00_C.sh $id &
    ./experimenten/core_01_C.sh $id &
    wait
    ./experimenten/core_02_C.sh $id &
    ./experimenten/core_03_C.sh $id &
    wait
    grep solved tmp/$id/log*.txt | wc --lines > tmp/$id/resultaat_$id.txt
    echo $id scores `cat tmp/$id/resultaat_$id.txt`
done
echo
for id in $*
do
    ./experimenten/core_00_D.sh $id &
    ./experimenten/core_01_D.sh $id &
    wait
    ./experimenten/core_02_D.sh $id &
    ./experimenten/core_03_D.sh $id &
    wait
    grep solved tmp/$id/log*.txt | wc --lines > tmp/$id/resultaat_$id.txt
    echo $id scores `cat tmp/$id/resultaat_$id.txt`
done
