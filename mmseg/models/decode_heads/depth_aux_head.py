"""
Copy-paste from torch.nn.Transformer, timm, with modifications:
"""
from re import I
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.decode_heads.daformer_head import DAFormerHead
import  torch
from torch import nn

from ..builder import HEADS

@HEADS.register_module()
class DadaDepthAuxBlock(BaseDecodeHead):
    def __init__(self, **kwargs):
        super(DadaDepthAuxBlock, self).__init__(
            input_transform='multiple_select', **kwargs)
        print('ctrl/model/decoder.py --> class DecoderAuxBlock -->  __init__()')

        self.models = nn.ModuleList()
        for in_chnl in self.in_channels:
            h1, h2 = in_chnl//4, in_chnl//16
            dec1 = nn.Conv2d(in_chnl, h1, kernel_size=1, stride=1, padding=0, bias=True)
            dec2 = nn.Conv2d(h1, h1, kernel_size=3, stride=1, padding=1, bias=True)
            dec3 = nn.Conv2d(h1, h2, kernel_size=1, stride=1, padding=0, bias=True)
            dec4 = nn.Conv2d(h2, in_chnl, kernel_size=1, stride=1, padding=0, bias=True)
            dec1.weight.data.normal_(0, 0.01)
            dec2.weight.data.normal_(0, 0.01)
            dec3.weight.data.normal_(0, 0.01)
            dec4.weight.data.normal_(0, 0.01)
            seq1 = nn.Sequential(dec1, nn.ReLU(inplace=True), dec2, nn.ReLU(inplace=True), dec3)
            seq2 = nn.Sequential(dec4, nn.ReLU(inplace=True))
            modlist = nn.ModuleList([seq1, seq2])
            self.models.append(modlist)
        
        in_chnls = kwargs.get("in_channels", [])
        kwargs["in_channels"] = [x//16 for x in in_chnls]
        self.segnet = DAFormerHead(**kwargs)

    def forward(self, x):
        # encoder
        x_dec3 = []
        x_dec4 = []
        j = 0
        for i in range(len(x)):
            if i in self.in_index:
                x_dec3.append(self.models[j][0](x[i]))
                x_dec4.append(self.models[j][1](x_dec3[-1]))
                j += 1
            else:
                x_dec3.append(None)
                x_dec4.append(torch.zeros_like(x[I]))
        out = self.segnet(x_dec3)
        return out, x_dec4