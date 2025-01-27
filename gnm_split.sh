#!/bin/bash 

if ["$1" = "true"];
then
    echo $1
    for n in sacson_labelled scand_labelled go_stanford_cropped_labelled cory_hall_labelled go_stanford2_labelled recon_labelled tartan_drive_labelled seattle_labelled cory_hall_cf_v1 cory_hall_cf_v2 go_stanford_cf_v1 go_stanford_cf_v2 scand_cf_v1 scand_cf_v2 sacson_cf_v1 sacson_cf_v2 go_stanford2_cf tartan_drive_cf seattle_cf recon_cf;
    do
        rm -r /home/noam/LLLwL/lcbc/data/data_splits/"$n"
    done  
fi

for n in sacson_labelled scand_labelled go_stanford_cropped_labelled cory_hall_labelled go_stanford2_labelled recon_labelled tartan_drive_labelled seattle_labelled cory_hall_cf_v1 cory_hall_cf_v2 go_stanford_cf_v1 go_stanford_cf_v2 scand_cf_v1 scand_cf_v2 sacson_cf_v1 sacson_cf_v2 go_stanford2_cf tartan_drive_cf seattle_cf recon_cf;
do
    python data_split.py -i /hdd/lcbc_datasets/"$n" -d "$n"
done

