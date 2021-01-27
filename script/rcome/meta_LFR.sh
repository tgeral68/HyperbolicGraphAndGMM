# UNIX script for launching experiments
#!/bin/bash

helpFunction()
{
    echo ""
    echo "Usage: $0 -l value1 -s value2 "
    echo -e "\t-l learning rate"
    echo -e "\t-s representation size"
    echo -e "\t-p prefix in log folder"
    echo -e "\t-c number of parallel launch"
}

min=0
max=8
prefix=""
parallel=3
while getopts "l:h:p:c:" opt
do
    case "$opt" in
            l ) min="$OPTARG" ;;
            h ) max="$OPTARG" ;;
            p ) prefix="$OPTARG" ;;
            c ) parallel="$OPTARG" ;;
    esac
done
dim=(2)
for d in "${dim[@]}"
    do
        nb_xp=`expr $max - $min`
        for i in $(seq $nb_xp)
            do
                dataset=`expr $i + $min` 
                echo sh script/rcome/LFR.sh -d "$d" -e "$dataset" -n 3 "$prefix" -c 0
                sh script/rcome/LFR.sh -d "$d" -e "$dataset" -n 3 -p "$prefix" -c 0 &
                #Launch by batch of parallel xps
                if [ $((i % parallel)) -eq 0 ]
                    then
                        wait
                fi
            done
    done