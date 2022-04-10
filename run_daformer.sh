#!/bin/bash
#SBATCH  --output=log/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=30G

source env/bin/activate

#python -m demo.image_demo demo/demo.png work_dirs/211108_1622_gta2cs_daformer_s0_7f24c/211108_1622_gta2cs_daformer_s0_7f24c.json work_dirs/211108_1622_gta2cs_daformer_s0_7f24c/latest.pth

#python tools/convert_datasets/gta.py data/gta --nproc 8
#python tools/convert_datasets/cityscapes.py data/cityscapes --nproc 8
#python tools/convert_datasets/synthia.py data/synthia/ --nproc 8

#sh tools/seg_test.sh work_dirs/local-basic/220330_1849_gta2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb5_s0_dcb1e_first_fulltraining_daformer

# SEG test
# python -m tools.seg_test work_dirs/local-basic/220330_1849_gta2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb5_s0_dcb1e_first_fulltraining_daformer/220330_1849_gta2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb5_s0_dcb1e.json work_dirs/local-basic/220330_1849_gta2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb5_s0_dcb1e_first_fulltraining_daformer/latest.pth --eval mIoU --show-dir work_dirs/local-basic/220330_1849_gta2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb5_s0_dcb1e_first_fulltraining_daformer/preds --opacity 1

# SEG train
# python run_experiments.py --config configs/daformer/gta2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb5_s0.py 

# PAN test
#python -m tools.pan_test  ./configs/panformer/panformer_r50_12e_coco_panoptic.py ./work_dirs/models/panoptic_segformer_r50_1x.pth --eval panoptic 
#python -m tools.pan_test  ./configs/panformer/panformer_r50_24e_coco_panoptic.py ./work_dirs/models/panoptic_segformer_r50_2x.pth --eval panoptic 
#python -m tools.pan_test  ./configs/panformer/panformer_r101_24e_coco_panoptic.py ./work_dirs/models/panoptic_segformer_r101_2x.pth --eval panoptic 
#python -m tools.pan_test ./configs/panformer/panformer_pvtb5_24e_coco_panoptic.py ./work_dirs/models/panoptic_segformer_pvtv2b5_2x.pth --eval panoptic 
#python -m tools.pan_test  ./configs/panformer/panformer_swinl_24e_coco_panoptic.py ./work_dirs/models/panoptic_segformer_swinl_2x.pth --eval panoptic 

# PAN train
# python -m tools.pan_train ./configs/panformer/panformer_r50_12e_coco_panoptic.py --deterministic