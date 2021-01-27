# UNIX script for launching experiments
#!/bin/bash

helpFunction()
{
    echo ""
    echo "Usage: $0 -l value1 -s value2 "
    echo -e "\t-l learning rate"
    echo -e "\t-s representation size"
    echo -e "\t-n number of experiments"
    echo -e "\t-c cuda device identifier"
    echo -e "\t-b batch size"
    echo -e "\t-p prefix"
    echo -e "\t-d Date in format mm-dd-yyyy if not filled today date $(date +'%m-%d-%Y')"
    exit 1 # Exit script after printing help
}

lr=0.05
size=2
nb_experiments=5
cuda_device="0"
batch_size=10
prefix=0
date_launched=$(date +'%m-%d-%Y')

while getopts "l:c:n:s:b:p:d:" opt
do
    case "$opt" in
            l ) lr="$OPTARG" ;;
            c ) cuda_device="$OPTARG" ;;
            n ) nb_experiments="$OPTARG" ;;
            s ) size="$OPTARG" ;;
            b ) batch_size="$OPTARG" ;;
            p ) prefix="$OPTARG" ;;
            d ) date_launched="$OPTARG" ;;
            ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
    esac
done





echo "Options for the following script : "
echo "\tScript name : $0"
echo "\tNumber of experiments : $nb_experiments"
echo "\tLearning rate : $lr"
echo "\tDimenssion of experiments : $size"
no_cuda="-1"
if [ "$cuda_device" = "$no_cuda" ]; then
    echo "\tTraining on CPU"
    cuda_arg=""
else
    echo "\tTraining on cuda device : $cuda_device"
    cuda_arg="--cuda"
fi
echo "\tBatch size : $batch_size"
echo "\tPrefix : $prefix"
echo "\tDate : $date_launched"
export OMP_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES="$cuda_device"

echo "\nStarting experiments \n"

for i in $(seq $nb_experiments)
    do
        echo "Experiment $i/$nb_experiments"
        echo CUDA_VISIBLE_DEVICES="$cuda_device" python experiment_script/rcome_learning.py --draw --loss-aggregation sum --distance-coef 1.0  --dataset football  --n-gaussian 12 --walk-lenght 30 --precompute-rw 10  --epoch 5 --epoch-embedding 1  --beta 0.05  --alpha 1 --gamma 0.01 --lr  "$lr" --negative-sampling 3  --context-size 2  $cuda_arg --embedding-optimizer radam  --em-iter 100 --id "$prefix$size"D-"$date_launched"-football-"$i" --size "$size"  --seed "$i" --batch-size "$batch_size"  --alpha 1.--beta .1 --epoch-embedding-init 5
        # CUDA_VISIBLE_DEVICES="$cuda_device" python experiment_script/rcome_learning.py --draw --loss-aggregation mean --distance-coef 1.0  --dataset football  --n-gaussian 12 --walk-lenght 30 --precompute-rw 5  --epoch  5 --O3-max-iter 10 --epoch-embedding 1  --beta 0.05  --alpha 1 --gamma 1. --lr  "$lr" --negative-sampling 20 --context-size 2  $cuda_arg --embedding-optimizer hsgd  --em-iter 100 --id "$prefix$size"D-"$date_launched"-football-"$i" --size "$size" --loss-aggregation mean --seed "$i" --batch-size "$batch_size"  --alpha 1. --beta .1 --epoch-embedding-init 5
        CUDA_VISIBLE_DEVICES="$cuda_device" python experiment_script/rcome_learning.py --same-context-embedding --draw --loss-aggregation sum --distance-coef 1.0  --dataset football  --n-gaussian 12 --walk-lenght 20 --precompute-rw 2 --epoch  5 --epoch-embedding  2  --beta 0.05  --lr  "$lr" --negative-sampling 5 --context-size 5  $cuda_arg --gamma .01 --embedding-optimizer radam  --em-iter 10 --id "$prefix$size"D-"$date_launched"-football-"$i" --size "$size"  --seed "$i" --batch-size "$batch_size"  --alpha 0.1 --beta .1 --epoch-embedding-init 5

    done

echo "Evaluation"

for i in $(seq $nb_experiments)
    do
        echo "Evaluation $i/$nb_experiments"

        CUDA_VISIBLE_DEVICES="$cuda_device" python evaluations_script/evaluation_unsupervised_poincare.py --id "$prefix$size"D-"$date_launched"-football-"$i" $cuda_arg
        CUDA_VISIBLE_DEVICES="$cuda_device" python evaluations_script/evaluation_unsupervised_poincare.py --id "$prefix$size"D-"$date_launched"-football-"$i" $cuda_arg --init
        CUDA_VISIBLE_DEVICES="$cuda_device" python evaluations_script/evaluation_classifier_poincare.py --id "$prefix$size"D-"$date_launched"-football-"$i" $cuda_arg
        CUDA_VISIBLE_DEVICES="$cuda_device" python evaluations_script/evaluation_supervised_poincare.py --id "$prefix$size"D-"$date_launched"-football-"$i" $cuda_arg --n-fold 5
    done