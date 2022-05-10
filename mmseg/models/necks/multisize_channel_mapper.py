import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule

from ..builder import NECKS


@NECKS.register_module()
class MultisizeChannelMapper(BaseModule):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(MultisizeChannelMapper, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        assert isinstance(out_channels, list)
        assert len(in_channels) == len(out_channels)
        self.convs = nn.ModuleList()
        for in_channel,out_channel in zip(in_channels,out_channels):
            self.convs.append(
                ConvModule(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=(kernel_size - 1) // 2,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.convs)
        outs = [self.convs[i](inputs[i]) for i in range(len(inputs))]
        return tuple(outs)
