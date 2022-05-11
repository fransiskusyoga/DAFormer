# DAFormer (with context-aware feature fusion) in Tab. 7

find_unused_parameters = True
norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/mit_b5.pth',
    backbone=dict(type='mit_b5', style='pytorch'),

    neck=dict(
        type='MultisizeChannelMapper',
        in_channels=[64, 128, 320, 512],
        kernel_size=1,
        out_channels=[256, 256, 256, 256], # 4 numbers should be the same
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32)
        ),
    decode_head=dict(
        type='DualFormerHead',
        in_channels=[256,256,256], # only recive list len 3 with all 3 numbers are the same
        #hw_lvl=[[128,128],[64,64],[32,32],[16,16]],
        in_index=[0, 2, 3], # must be list of lenght 3
        channels=256,
        #dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        num_classes=19,
        align_corners=False,
        num_head=8,
        num_layers=6,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
