#!/bin/bash 


# if [$1 = "true"];
# then
#     echo $1
#     for n in sacson go_stanford_cropped go_stanford2 cory_hall scand tartan_drive recon seattle;
#     do
#     for i in left right forward stop;
#         do
#             rm -r /home/noam/LLLwL/lcbc/data/data_splits/"$n"_atomic_"$i"
#         done
#     done  
# fi
# for n in sacson go_stanford_cropped go_stanford2 cory_hall scand tartan_drive recon seattle;
# do
#    for i in left right forward stop;
#     do
#         if [ "$i" == "left" ];
#         then
#             python data_split.py -i /home/noam/LLLwL/datasets/atomic_dataset_fixed/$n/turn_$i/ -d "$n"_atomic_"$i"
#         fi
#         if [ "$i" == "right" ];
#         then
#             python data_split.py -i /home/noam/LLLwL/datasets/atomic_dataset_fixed/$n/turn_$i/ -d "$n"_atomic_"$i"
#         fi
#         if [ "$i" == "forward" ];
#         then
#             python data_split.py -i /home/noam/LLLwL/datasets/atomic_dataset_fixed/$n/go_$i/ -d "$n"_atomic_"$i"
#         fi
#         if [ "$i" == "stop" ];
#         then
#             python data_split.py -i /home/noam/LLLwL/datasets/atomic_dataset_fixed/$n/$i/ -d "$n"_atomic_"$i"
#         fi
#     done
# done

if ["$1" = "true"];
then
    echo $1
    for n in sacson_labelled scand_labelled go_stanford_cropped_labelled cory_hall_labelled;
    do
        rm -r /home/cglossop/vlm-guidance/data/data_splits/"$n"
    done  
fi

for n in sacson_labelled scand_labelled go_stanford_cropped_labelled cory_hall_labelled;
do
    python data_split.py -i /home/cglossop/lcbc_datasets/"$n" -d "$n"
done

