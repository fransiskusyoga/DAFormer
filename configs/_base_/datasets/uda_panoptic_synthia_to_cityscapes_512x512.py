# dataset settings
dataset_type = 'CityscapesDataset'
data_root = 'data/cityscapes/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
synthia_train_pipeline = [
    dict(type='LoadImageFromFilePanUDA'),
    dict(type='LoadAnnotationsPanUDA'),
    dict(type='ResizePanUDA', img_scale=(1280, 760)),
    dict(type='RandomCropPanUDA', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlipPanUDA', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='NormalizePanUDA', **img_norm_cfg),
    dict(type='PadPanUDA', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
cityscapes_train_pipeline = [
    dict(type='LoadImageFromFilePanUDA'),
    dict(type='LoadAnnotationsPanUDA'),
    dict(type='ResizePanUDA', img_scale=(1024, 512)),
    dict(type='RandomCropPanUDA', crop_size=crop_size),
    dict(type='RandomFlipPanUDA', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='NormalizePanUDA', **img_norm_cfg),
    dict(type='PadPanUDA', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 512),
        # MultiScaleFlipAug is disabled by not providing img_ratios and
        # setting flip=False
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='UDADataset',
        source=dict(
            type='SynthiaDataset_panoptic',
            data_root='data/synthia/',
            img_dir='RGB',
            seg_map_dir='GT/LABELS',
            pan_map_dir='GT/panoptic-labels-crowdth-0/synthia_panoptic',
            ann_file='GT/panoptic-labels-crowdth-0/synthia_panoptic.json',
            pipeline=synthia_train_pipeline),
        target=dict(
            type='CityscapesDataset_panoptic',
            data_root='data/cityscapes/',
            img_dir='leftImg8bit/train',
            seg_map_dir='gtFine/train',
            pan_map_dir='gtFine/cityscapes_panoptic_synthia_to_cityscapes_16cls_train_trainId',
            ann_file='gtFine/cityscapes_panoptic_synthia_to_cityscapes_16cls_train_trainId.json',
            pipeline=cityscapes_train_pipeline)),
    val=dict(
        type='CityscapesDataset',
        data_root='data/cityscapes/',
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=test_pipeline),
    test=dict(
        type='CityscapesDataset',
        data_root='data/cityscapes/',
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=test_pipeline))
