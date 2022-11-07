#!/bin/bash

# python3 train-finetune.py --train_folder=../../data/spacenetV2_splits/train_mask_sharp/ --weight=weights/PPLinknet_psp64_osm_building_manual_phase1_8375.th --name=PPLinknet_psp64_osm_building_manual_phase1_8375_spacenet_sharp_ft01.th --shape=512 --batchsize=9

python3 test_mean_std.py --weight=weights/PPLinknet_psp64_osm_building_manual_phase1_8375_spacenet_sharp_ft01.th --jpg_folder=../../data/spacenetV2_splits/test_mask_sharp/ --threshold=2

python3 test_mean_std.py --weight=weights/PPLinknet_psp64_spacenet_sharp_01.th --jpg_folder=../../data/spacenetV2_splits/test_mask_sharp/

python3 test_mean_std.py --weight=weights/PPLinknet_psp64_spacenet_sharp_01.th --jpg_folder=../../data/spacenetV2_splits/test_mask_sharp/ --threshold=2

# python3 train.py --train_folder=../../data/grab-deep-globe/building_osm_dataset_manual/ --name=PPLinknet_psp64_osm_building_manual_phase1_8375.th --batchsize=8