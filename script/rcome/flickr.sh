# UNIX script for launching experiments
#!/bin/bash


helpFunction()
{
    echo ""
    echo "Usage: $0 -arg1 value1 -arg2 value2 "
    echo "\t-l learning rate"
    echo "\t-s representation size"
    echo "\t-n number of experiments"
    echo "\t-c cuda device identifier"
    echo "\t-b batch size"
    echo "\t-p prefix"
    echo "\t-d Date in format mm-dd-yyyy if not filled today date $(date +'%m-%d-%Y')"
    exit 1 # Exit script after printing help
}

lr=.08
size=5
nb_experiments=5
cuda_device="0"
batch_size=256
prefix=0
date_launched=$(date +'%m-%d-%Y')
i=0
while getopts "l:c:n:s:b:p:d:i:" opt
do
    case "$opt" in
            l ) lr="$OPTARG" ;;
            c ) cuda_device="$OPTARG" ;;
            n ) nb_experiments="$OPTARG" ;;
            s ) size="$OPTARG" ;;
            b ) batch_size="$OPTARG" ;;
            p ) prefix="$OPTARG" ;;
            d ) date_launched="$OPTARG" ;;
            i ) i="$OPTARG" ;;
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


export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES="$cuda_device"

echo "\nStarting experiments \n"

# for i in $(seq $nb_experiments)
#     do  
echo "Experiment $i/$nb_experiments"
echo CUDA_VISIBLE_DEVICES="$cuda_device" python experiment_script/rcome_learning.py --loss-aggregation sum --distance-coef 1.0  --dataset flickr  --n-gaussian 195 --walk-lenght 40 --precompute-rw 3  --epoch 10 --epoch-embedding 1  --lr  "$lr" --negative-sampling 5  --context-size 10  $cuda_arg --embedding-optimizer radam  --em-iter 50 --id "$prefix$size"D-"$date_launched"-flickr-"$i" --size "$size"  --seed "$i" --batch-size "$batch_size" --alpha 1. --beta 1. --gamma 0.01 --epoch-embedding-init 20
CUDA_VISIBLE_DEVICES="$cuda_device" python experiment_script/rcome_learning.py --loss-aggregation sum --distance-coef 1.0  --dataset flickr  --n-gaussian 195 --walk-lenght 40 --precompute-rw 3 --epoch 10 --epoch-embedding 1 --lr  "$lr" --negative-sampling 5  --context-size 10  $cuda_arg --embedding-optimizer radam  --em-iter 50 --id "$prefix$size"D-"$date_launched"-flickr-"$i" --size "$size"  --seed "$i" --batch-size "$batch_size" --alpha 1. --beta 1. --gamma 0.01 --epoch-embedding-init 20 --reset-em
    # done

# echo "Evaluation"

# for i in $(seq $nb_experiments)
#     do
echo "Evaluation $i/$nb_experiments"
# CUDA_VISIBLE_DEVICES="$cuda_device" python evaluations_script/evaluation_unsupervised_poincare.py  --id "$prefix$size"D-"$date_launched"-flickr-"$i" $cuda_arg
CUDA_VISIBLE_DEVICES="$cuda_device" python evaluations_script/evaluation_classifier_poincare.py  --id "$prefix$size"D-"$date_launched"-flickr-"$i" $cuda_arg
CUDA_VISIBLE_DEVICES="$cuda_device" python evaluations_script/evaluation_supervised_poincare.py  --id "$prefix$size"D-"$date_launched"-flickr-"$i" $cuda_arg
    # done