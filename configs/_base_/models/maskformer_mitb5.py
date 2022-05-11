# DAFormer (with context-aware feature fusion) in Tab. 7

find_unused_parameters = True
norm_cfg = dict(type='BN', requires_grad=True)
_m_dim_ = 256
_m_dim_half_ = _m_dim_//2
_m_feed_ratio_ = 4
_m_feed_dim_ = _m_feed_ratio_*_m_dim_
_m_num_levels_=4

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
        type='MaskFormerHead',
        in_channels=[256, 256, 256, 256],
        in_index=[0, 1, 2, 3],
        channels=256,
        norm_cfg=norm_cfg,
        num_classes=19,
        transformer=dict(
            type='Deformable_Transformer',
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=_m_dim_,
                        num_levels=_m_num_levels_,
                         ),
                    feedforward_channels=_m_feed_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict( # this decodeer is not used you can define it as you want. as small as possible
                type='DeformableDetrTransformerDecoder',
                num_layers=1,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_m_dim_,
                            num_heads=2,
                            dropout=0.1),
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=_m_dim_,
                            num_levels=_m_num_levels_,
                        )
                    ],
                    feedforward_channels=_m_feed_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=_m_dim_half_,
            normalize=True,
            offset=-0.5),
        mask_head=dict(type='MaskHead',d_model=_m_dim_,nhead=8,num_decoder_layers=6,self_attn=True),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
        ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
