#!/bin/bash
#SBATCH  --output=log/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=30G

source env/bin/activate
#python -m demo.image_demo demo/demo.png work_dirs/211108_1622_gta2cs_daformer_s0_7f24c/211108_1622_gta2cs_daformer_s0_7f24c.json work_dirs/211108_1622_gta2cs_daformer_s0_7f24c/latest.pth
#cp -R -v /srv/beegfs02/scratch/uda2022/data/datasets/synthia/ /scratch_net/biwidl312_second/ywibowo/datasets

#python tools/convert_datasets/gta.py data/gta --nproc 8
#python tools/convert_datasets/cityscapes.py data/cityscapes --nproc 8
#python tools/convert_datasets/synthia.py data/synthia/ --nproc 8

#python run_experiments.py --config configs/daformer/gta2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb5_s0.py
#python run_experiments.py --config configs/daformer/synthia2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb5_s0.py 

#python run_experiments.py --exp 7

#python run_experiments.py --config configs/daformer/gta2cs_backbone_resnet.py 
#python run_experiments.py --config configs/daformer/synthia2cs_backbone_resnet.py 
#python run_experiments.py --config configs/daformer/gta2cs_backbone_resnet101.py 
#python run_experiments.py --config configs/daformer/synthia2cs_backbone_resnet101.py 
#python run_experiments.py --config configs/daformer/gta2cs_backbone_pvtb5.py 
#python run_experiments.py --config configs/daformer/synthia2cs_backbone_pvtb5.py 
#python run_experiments.py --config configs/daformer/gta2cs_depthformer_mitb5.py 
python run_experiments.py --config configs/daformer/gta2cs_maskformer_mitb5.py 

# python run_experiments.py --config configs/daformer/gta2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb5_s0.py
# python run_experiments.py --config configs/daformer/gta2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_r101_s0.py
# python run_experiments.py --config configs/daformer/gta2cs_uda_warm_fdthings_rcs_croppl_a999_dlv3p_r101_s0.py
# python run_experiments.py --config configs/daformer/gta2cs_uda_warm_fdthings_rcs_croppl_a999_dlv3p_mitb5_s0.py


#XDG_CACHE_HOME=/scratch_net/biwidl312/ywibowo/.cache/ pip install mmcv-full==1.3.7 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html
# pip install torch==1.10.2+cu111 torchvision==0.11.3+cu111 -f https://download.pytorch.org/whl/torch_stable.html
# pip cache purge
# pip install -r requirements.txt
# pip cache purge
# pip install mmcv-full==1.3.7
# cd /scratch_net/biwidl312/ywibowo/mmcv-1.3.7 && MMCV_WITH_OPS=1 pip install -e .
# pip install timm torch==1.10.2+cu113 torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/torch_stable.html
# pip install -r requirements.txt


