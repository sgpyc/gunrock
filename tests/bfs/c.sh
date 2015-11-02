#!/bin/bash

TESTNAME=soc-Livejournal1
for i in {0..128}
do
    ./a.sh > ${TESTNAME}_$i.txt
    printf "${TESTNAME}_$i "
    grep diff ${TESTNAME}_$i.txt
    grep Vertex ${TESTNAME}_$i.txt | grep reference
done
