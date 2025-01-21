#!/bin/bash 

if ["$1" = "true"];
then
    echo $1
    for n in sacson_cf scand_cf go_stanford_cf cory_hall_cf;
    do
        rm -r /home/noam/LLLwL/lcbc/data/data_splits/"$n"
    done  
fi

for n in sacson_cf scand_cf go_stanford_cf cory_hall_cf;
do
    python data_split.py -i /home/noam/LLLwL/lcbc/data/data_annotation/cf_dataset_v2/"$n" -d "$n"
done

