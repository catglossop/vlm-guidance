#!/bin/bash 

if [$1 = "true"];
then
    echo $1
    for n in sacson go_stanford_cropped go_stanford2 cory_hall scand tartan_drive recon seattle;
    do
    for i in left right forward stop;
        do
            rm -r /home/noam/LLLwL/lcbc/data/data_splits/"$n"_atomic_"$i"
            rm -r /home/noam/LLLwL/lcbc/data/data_splits/"$n"_atomic_adjust_"$i"
        done
    done  
fi
for n in sacson go_stanford_cropped go_stanford2 cory_hall scand tartan_drive recon seattle;
do
   for i in left right forward stop;
    do
        if [ "$i" == "left" ];
        then
            python data_split.py -i /home/noam/LLLwL/datasets/atomic_dataset_finer/$n/turn_$i/ -d "$n"_atomic_"$i"
            python data_split.py -i /home/noam/LLLwL/datasets/atomic_dataset_finer/$n/adjust_$i/ -d "$n"_atomic_adjust_"$i"
        fi
        if [ "$i" == "right" ];
        then
            python data_split.py -i /home/noam/LLLwL/datasets/atomic_dataset_finer/$n/turn_$i/ -d "$n"_atomic_"$i"
            python data_split.py -i /home/noam/LLLwL/datasets/atomic_dataset_finer/$n/adjust_$i/ -d "$n"_atomic_adjust_"$i"
        fi
        if [ "$i" == "forward" ];
        then
            python data_split.py -i /home/noam/LLLwL/datasets/atomic_dataset_finer/$n/go_$i/ -d "$n"_atomic_"$i"
        fi
        if [ "$i" == "stop" ];
        then
            python data_split.py -i /home/noam/LLLwL/datasets/atomic_dataset_finer/$n/$i/ -d "$n"_atomic_"$i"
        fi
    done
done


