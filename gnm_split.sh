#!/bin/bash 

if ["$1" = "true"];
then
    echo $1
    for n in sacson_labelled scand_labelled go_stanford_cropped_labelled cory_hall_labelled;
    do
        rm -r /home/noam/LLLwL/lcbc/data/data_splits/"$n"
    done  
fi

for n in sacson_labelled scand_labelled go_stanford_cropped_labelled cory_hall_labelled;
do
    python data_split.py -i /home/noam/LLLwL/lcbc/data/data_annotation/lcbc_datasets_backup/"$n" -d "$n"
done

