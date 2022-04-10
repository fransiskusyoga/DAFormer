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
# python -m tools.seg_test work_dirs/211108_1622_gta2cs_daformer_s0_7f24c/211108_1622_gta2cs_daformer_s0_7f24c.json work_dirs/211108_1622_gta2cs_daformer_s0_7f24c/latest.pth --eval mIoU --show-dir work_dirs/211108_1622_gta2cs_daformer_s0_7f24c/preds --opacity 1

# SEG train
rm -rf mmseg/datasets/pipelines mmseg/datasets/builder.py mmseg/datasets/custom.py mmseg/datasets/dataset_wrappers.py
cp -R mmseg/datasets/pipelines_seg mmseg/datasets/pipelines
cp mmseg/datasets/builder_seg.py mmseg/datasets/builder.py
cp mmseg/datasets/custom_seg.py mmseg/datasets/custom.py
cp mmseg/datasets/dataset_wrappers_seg.py mmseg/datasets/dataset_wrappers.py
echo "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
python run_experiments.py --config configs/daformer/gta2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb5_s0.py 

# PAN test
rm -rf mmseg/datasets/pipelines mmseg/datasets/builder.py mmseg/datasets/custom.py mmseg/datasets/dataset_wrappers.py
cp -R mmseg/datasets/pipelines_pan mmseg/datasets/pipelines
cp mmseg/datasets/builder_pan.py mmseg/datasets/builder.py
cp mmseg/datasets/custom_pan.py mmseg/datasets/custom.py
cp mmseg/datasets/dataset_wrappers_pan.py mmseg/datasets/dataset_wrappers.py
echo "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"
python -m tools.pan_test  ./configs/panformer/panformer_r50_12e_coco_panoptic.py ./work_dirs/models/panoptic_segformer_r50_1x.pth --eval panoptic 
echo "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"
python -m tools.pan_test  ./configs/panformer/panformer_r50_24e_coco_panoptic.py ./work_dirs/models/panoptic_segformer_r50_2x.pth --eval panoptic
echo "DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD" 
python -m tools.pan_test  ./configs/panformer/panformer_r101_24e_coco_panoptic.py ./work_dirs/models/panoptic_segformer_r101_2x.pth --eval panoptic 
echo "EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE"
python -m tools.pan_test ./configs/panformer/panformer_pvtb5_24e_coco_panoptic.py ./work_dirs/models/panoptic_segformer_pvtv2b5_2x.pth --eval panoptic 
echo "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF"
python -m tools.pan_test  ./configs/panformer/panformer_swinl_24e_coco_panoptic.py ./work_dirs/models/panoptic_segformer_swinl_2x.pth --eval panoptic 

# PAN train
# python -m tools.pan_train ./configs/panformer/panformer_r50_12e_coco_panoptic.py --deterministic