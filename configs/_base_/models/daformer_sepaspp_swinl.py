# DAFormer (with context-aware feature fusion) in Tab. 7

_base_ = ['daformer_conv1_mitb5.py']

norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    pretrained='./pretrained/swinl.pth',
    backbone=dict(
        type='SwinTransformer',
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        ape=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        use_checkpoint=False,
        style=None),
    # neck=dict(
    #     type='MultisizeChannelMapper',
    #     in_channels=[256, 512, 1024, 2048],
    #     kernel_size=1,
    #     out_channels=[64, 128, 320, 512],
    #     act_cfg=None,
    #     norm_cfg=dict(type='GN', num_groups=32)
    #     ),
    decode_head=dict(
        in_channels=[192, 384, 768, 1536],
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
