import torch
from torch import nn
from mmcv.runner import force_fp32
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmseg.models.builder import HEADS
from mmseg.models.utils.builder import build_transformer
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class MaskFormerHead(BaseDecodeHead):
    """
    Head of Panoptic SegFormer

    Code is modified from the `official github repo
    <https://github.com/open-mmlab/mmdetection>`_.

    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
    """
    def __init__(
            self,
            transformer=None,
            use_argmax=False,
            positional_encoding= dict(type='SinePositionalEncoding',
                                     num_feats=128,
                                     normalize=True),
            mask_head=dict(
                type='TransformerHead',  # mask decoder for stuff
                d_model=256,
                nhead=8,
                num_decoder_layers=6),
            **kwargs):
        super(MaskFormerHead, self).__init__(input_transform='multiple_select', **kwargs)
        self.use_argmax = use_argmax
        self.fp16_enabled = False

        self.num_decode = mask_head['num_decoder_layers']
        
        self.positional_encoding = build_positional_encoding(positional_encoding)
        self.transformer = build_transformer(transformer)
        self.mask_head = build_transformer(mask_head)
        
        self.embed_dims = self.transformer.embed_dims
        self.query_embedding = nn.Embedding(2,self.embed_dims * 2) # this variable is useless, remove it
        self.query_mask = nn.Embedding(self.num_classes,
                                    self.embed_dims * 2)
        self.count = 0

    @force_fp32(apply_to=('x'))
    def forward(self, x):
        # Step 1: Mask and pos encoding
        # Create mask for different input image size in the same batch
    
        batch_size = x[0].size(0)
        hw_lvl = []
        mlvl_masks = []
        mlvl_positional_encodings = []
        print([a.shape for a in x])
        for feat in x:
            hw_lvl.append(
                feat.shape[-2:])
            mlvl_masks.append(
                x[0].new_zeros((batch_size,*feat.shape[-2:])).to(torch.bool))
            print(mlvl_masks[-1].shape)
            mlvl_positional_encodings.append(
                self.positional_encoding(mlvl_masks[-1]))
        
        # Step 2: Run The transformer (get bbox and intermidiate vals for mask head)
        # Fetch the querry mebedding
        query_embeds = self.query_embedding.weight
        (memory, memory_pos, memory_mask, query_pos), hs, init_reference, inter_references, \
        enc_outputs_class, enc_outputs_coord = self.transformer(
            x,
            mlvl_masks,
            query_embeds,
            mlvl_positional_encodings,
            reg_branches=None,  # noqa:E501
            cls_branches=None  # noqa:E501
        )
        # The intermidiate values are packed to varaible args_tuple
        # we should feed these to mask deocder.
        memory = memory.permute(1, 0, 2)
        query = hs[-1].permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        memory_pos = memory_pos.permute(1, 0, 2)

        len_last_feat = hw_lvl[-1][0] * hw_lvl[-1][1]
        memory      = memory[:, :-len_last_feat, :]
        memory_mask = memory_mask[:, :-len_last_feat]
        memory_pos  = memory_pos[:, :-len_last_feat, :]

        # -fetch stuff query
        query_mask, query_mask_pos = torch.split(self.query_mask.weight,
                                                   self.embed_dims,
                                                   dim=1)
        query_mask_pos = query_mask_pos.unsqueeze(0).expand(batch_size, -1, -1)
        query_mask = query_mask.unsqueeze(0).expand(batch_size, -1, -1)

        mask, mask_inter, query_inter = self.mask_head(
            memory, memory_mask, None, query_mask, None, query_mask_pos, hw_lvl=hw_lvl)

        print("Out of memory problem")
        assert False
        return mask

       
        