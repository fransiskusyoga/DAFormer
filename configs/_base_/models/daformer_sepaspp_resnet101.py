# DAFormer (with context-aware feature fusion) in Tab. 7

_base_ = ['daformer_conv1_mitb5.py']

norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    pretrained='torchvision://resnet101',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        #frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch'),
    # neck=dict(
    #     type='ChannelMapper',
    #     in_channels=[256, 512, 1024, 2048],
    #     kernel_size=1,
    #     out_channels=[64, 128, 320, 512],
    #     act_cfg=None,
    #     norm_cfg=dict(type='GN', num_groups=32)
    #     ),
    decode_head=dict(
        in_channels=[256, 512, 1024, 2048],
        in_index=[0, 1, 2, 3],
        decoder_params=dict(
            fusion_cfg=dict(
                _delete_=True,
                type='aspp',
                sep=True,
                dilations=(1, 6, 12, 18),
                pool=False,
                act_cfg=dict(type='ReLU'),
                norm_cfg=norm_cfg))))
