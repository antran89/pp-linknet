#!/bin/bash

# python3 train.py --train_folder=../../data/grab-deep-globe/train --name=PPLinknet_psp64_phase1_sea_205660.th --batchsize=8

python3 train-finetune.py --train_folder=../../data/grab-deep-globe/train_deepglobeall/ --weight=weights/PPLinknet_psp64_phase1_sea_205660.th --name=PPLinknet_psp64_phase1_sea_205660_ft_deepglobeall_01.th --batchsize=8

python3 test_mean_std.py --weight=weights/PPLinknet_psp64_phase1_sea_205660_ft_deepglobeall_01.th --jpg_folder=../../data/grab-deep-globe/valid/ --threshold=2