_base_ = [
    'deeplabv3plus_r50-d8.py'
]

model= dict(
    pretrained= "pretrained/mit_b5.pth",
    backbone= dict(
        _delete_= True,
        type= "mit_b5"
    ),
    neck= dict(
        type= "SegFormerAdapter"
    ),
    decode_head= dict(
        c1_in_channels= 64,
        in_channels= 512
    )
)