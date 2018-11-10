#!/bin/bash

#get all execution files in ./bin
files=(./bin/*)
#split file names into arr
arr=$(echo $files | tr " " "\n")
max_ver_num="$"

exe_file=${arr[0]}
#iterate over all file names to get the largest version number
for x in $arr
do
    output=$(grep -o "[0-9]\.[0-9]" <<<"$x")
    if [ "$output" \> "$max_ver_num" ]; then
        exe_file=$x
    fi
done

OPTIONS=""
OPTIONS="$OPTIONS --omp-threads=32 --omp-runs=5 --num-runs=0 --quick --validation=each"
#OPTIONS="$OPTIONS --source=8922 --sink=8923"
OPTIONS="$OPTIONS --neighbor-select=any"
OPTIONS="$OPTIONS --iter-stats"
OPTIONS="$OPTIONS --relabeling-interval=100" && MARKS="i100_a"

OPTION[ 0]="$OPTIONS --merge-push-relabel=false --active-vertices=false --use-atomic=false --use-residual=false" && MARK[ 0]="mvar" 
OPTION[ 1]="$OPTIONS --merge-push-relabel=false --active-vertices=false --use-atomic=true  --use-residual=false" && MARK[ 1]="mvAr" 
OPTION[ 2]="$OPTIONS --merge-push-relabel=false --active-vertices=true  --use-atomic=false --use-residual=false" && MARK[ 2]="mVar" 
OPTION[ 3]="$OPTIONS --merge-push-relabel=false --active-vertices=true  --use-atomic=true  --use-residual=false" && MARK[ 3]="mVAr" 
OPTION[ 4]="$OPTIONS --merge-push-relabel=true  --active-vertices=false --use-atomic=false --use-residual=false" && MARK[ 4]="Mvar" 
OPTION[ 5]="$OPTIONS --merge-push-relabel=true  --active-vertices=false --use-atomic=true  --use-residual=false" && MARK[ 5]="MvAr" 
OPTION[ 6]="$OPTIONS --merge-push-relabel=true  --active-vertices=true  --use-atomic=false --use-residual=false" && MARK[ 6]="MVar" 
OPTION[ 7]="$OPTIONS --merge-push-relabel=true  --active-vertices=true  --use-atomic=true  --use-residual=false" && MARK[ 7]="MVAr" 
OPTION[ 8]="$OPTIONS --merge-push-relabel=false --active-vertices=false --use-atomic=false --use-residual=true " && MARK[ 8]="mvaR" 
OPTION[ 9]="$OPTIONS --merge-push-relabel=false --active-vertices=false --use-atomic=true  --use-residual=true " && MARK[ 9]="mvAR" 
OPTION[10]="$OPTIONS --merge-push-relabel=false --active-vertices=true  --use-atomic=false --use-residual=true " && MARK[10]="mVaR" 
OPTION[11]="$OPTIONS --merge-push-relabel=false --active-vertices=true  --use-atomic=true  --use-residual=true " && MARK[11]="mVAR" 
OPTION[12]="$OPTIONS --merge-push-relabel=true  --active-vertices=false --use-atomic=false --use-residual=true " && MARK[12]="MvaR" 
OPTION[13]="$OPTIONS --merge-push-relabel=true  --active-vertices=false --use-atomic=true  --use-residual=true " && MARK[13]="MvAR" 
OPTION[14]="$OPTIONS --merge-push-relabel=true  --active-vertices=true  --use-atomic=false --use-residual=true " && MARK[14]="MVaR" 
OPTION[15]="$OPTIONS --merge-push-relabel=true  --active-vertices=true  --use-atomic=true  --use-residual=true " && MARK[15]="MVAR" 

#put OS and Device type here
EXCUTION=$exe_file
DATADIR="/data/Taxi_Datasets/gunrock/"

NAME[ 0]="wrong_graph_GPU" && SOURCE[0]="8922" && SINK[0]="8923"
NAME[ 1]="small"           && SOURCE[1]="0"    && SINK[1]="1"
NAME[ 2]="larger"          && SOURCE[2]="35688"    && SINK[2]="35689"
NAME[ 3]="largest"         && SOURCE[3]="213360"    && SINK[3]="213361"

for k in 0
do
    #put OS and Device type here
    SUFFIX="ubuntu18.04_GV100x1"
    LOGDIR=eval/$SUFFIX
    mkdir -p $LOGDIR

    for i in 0
    do
        for j in {0..15}
        do
            echo $EXCUTION market $DATADIR/${NAME[$i]}.mtx ${OPTION[$j]} --source=${SOURCE[$i]} --sink=${SINK[$i]} --jsondir=$LOGDIR "> $LOGDIR/${NAME[$i]}_${MARKS}${MARK[$j]}.txt 2>&1"
                 $EXCUTION market $DATADIR/${NAME[$i]}.mtx ${OPTION[$j]} --source=${SOURCE[$i]} --sink=${SINK[$i]} --jsondir=$LOGDIR  > $LOGDIR/${NAME[$i]}_${MARKS}${MARK[$j]}.txt 2>&1
            #sleep 30
        done
    done
done

