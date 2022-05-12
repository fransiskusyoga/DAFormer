"""
Copy-paste from torch.nn.Transformer, timm, with modifications:
"""
from audioop import bias
import copy
from typing import Optional, List
from mmseg.models.decode_heads.decode_head import BaseDecodeHead

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from functools import partial
import math
from mmcv.runner import force_fp32

from ..builder import HEADS

count = 0


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        self.fp16_enabled = False
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)

        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    @force_fp32(apply_to=('x', ))
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MaskAttention(nn.Module):
    def __init__(self,
                 dim,
                 n_channels,
                 kv_reduct =16,
                 q_reduct =4,
                 emb_dim=128,
                 n_heads = 4,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0,
                 proj_drop=0
                 ):
        super().__init__()
        total_dim = n_channels*dim
        self.total_dim = total_dim
        self.emb_dim = emb_dim
        self.dim = dim
        self.n_heads = n_heads
        self.scale = qk_scale or 1.0/math.sqrt(emb_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.conv_q = nn.Conv2d(
            total_dim, n_heads*emb_dim, q_reduct, stride=q_reduct, padding='valid', bias=qkv_bias)
        self.conv_k = nn.Conv2d(
            total_dim, n_heads*emb_dim, kv_reduct, stride=kv_reduct, padding='valid', bias=qkv_bias)
        self.conv_v = nn.Conv2d(
            total_dim, n_heads*total_dim, kv_reduct, stride=kv_reduct, padding='valid', bias=qkv_bias)
        self.aggregator = nn.Conv2d(
            n_heads, 1, 1, bias=False) #kernel=1 output=1 channel
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            
    def forward(self, x, hw_lvl):
        # Reshape flaten input to image, input shape x=[B,L,C] 
        # B=batchsize, L=total pixels from mutiple levels, C=channel dimension
        B,_,C = x.shape
        x = x.permute(0,2,1) # [B,C,L]
        wedge_curr = 0
        imgs = []
        for i in range(len(hw_lvl)):
            wedge_next = wedge_curr + hw_lvl[i][0] * hw_lvl[i][1]

            img = x[:, :, wedge_curr:wedge_next].reshape(B,C,*hw_lvl[i]) # [B,C,H_i,W_i]
            img = F.interpolate(img, size=hw_lvl[0], mode='bilinear') # [B,C,H_0,W_0]
            imgs.append(img)

            wedge_curr = wedge_next

        all_img = torch.cat(imgs, 1) # [B,TD,H_0,W_0]

        # Main Calculation
        # initial tranformation layer
        # L0 = H_0 * W_0 / q_reduct / q_reduct
        # LR0 =  H_0 * W_0 / kv_reduct / kv_reduct
        ED = self.emb_dim
        NH = self.n_heads
        TD = self.total_dim
        k = self.conv_k(all_img).reshape(B,NH,ED,-1) # [B,NH,ED,LR0]
        q = self.conv_q(all_img).reshape(B,NH,ED,-1).permute(0,1,3,2).contiguous() # [B,NH,L0,ED]
        v = self.conv_v(all_img).reshape(B,NH,TD,-1).permute(0,1,3,2).contiguous() # [B,NH,LR0,TD]

        # attention calculation
        attn = (q @ k)* self.scale # [B,NH,L0,LR0] potentially the biggest object
        attn = self.attn_drop(attn.softmax(dim=-1))
        attn = attn @ v # [B,NH,L0,TD] potentially the biggest object

        # Agregatting info from different head
        attn = self.aggregator(attn).squeeze(1) # [B,L0,TD]
        attn = attn.permute(0,2,1).contiguous() # [B,TD,L0]

        # Make it the simillar size with the input
        flat_imgs = []
        for i in range(len(hw_lvl)):
            flat_img = attn[:,i*self.dim:(i+1)*self.dim,:].reshape(B,self.dim,*hw_lvl[0]) # [B,C,H_0,W_0]
            flat_img = F.interpolate(img, size=hw_lvl[i], mode='bilinear') # [B,C,H_i,W_i]
            flat_img = flat_img.reshape(B,self.dim,-1) # [B,C,H_i*W_i]
            flat_imgs.append(flat_img)

        all_flat_img = torch.cat(flat_imgs, -1).permute(0,2,1).contiguous() # [B,L,C]

        result = self.proj_drop(all_flat_img)
        return result

class QueryAttention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=2,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.fp16_enabled = False
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @force_fp32(apply_to=('query', 'key', 'value'))
    def forward(self, query, key, value, hw_lvl):
        B, N, C = query.shape
        _, L, _ = key.shape
        #print('query, key, value', query.shape, value.shape, key.shape)
        q = self.q(query).reshape(B, N,
                                  self.num_heads, C // self.num_heads).permute(
                                      0, 2, 1,
                                      3).contiguous()  #.permute(2, 0, 3, 1, 4)  # B, NH, N, C/NH
        k = self.k(key).reshape(B, L,
                                self.num_heads, C // self.num_heads).permute(
                                    0, 2, 1,
                                    3).contiguous()  #.permute(2, 0, 3, 1, 4)  # B, NH, L, C/NH

        v = self.v(value).reshape(B, L,
                                  self.num_heads, C // self.num_heads).permute(
                                      0, 2, 1,
                                      3).contiguous()  #.permute(2, 0, 3, 1, 4) # B, NH, L, C/NH

        attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale # B, NH, N, L
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn) 

        x = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

# AttentionTail is a cheap implementation that can make mask decoder 1 layer deeper.
class AttentionTail(nn.Module): 
    def __init__(self,
                 dim,
                 num_heads=2,
                 qkv_bias=False,
                 qk_scale=None,
                 n_channels=4,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.fp16_enabled = False
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)

        linear_l = nn.Sequential(
            nn.Linear(self.num_heads, self.num_heads),
            nn.ReLU(),
        )
        self.linear_l = _get_clones(linear_l,n_channels)
        self.linear = nn.Sequential(
            nn.Linear(self.num_heads * n_channels, 1),
            nn.ReLU(),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @force_fp32(apply_to=('query', 'key'))
    def forward(self, query, key, hw_lvl=None):
        B, N, C = query.shape
        _, L, _ = key.shape
        #print('query, key, value', query.shape, value.shape, key.shape)
        q = self.q(query).reshape(B, N,
                                  self.num_heads, C // self.num_heads).permute(
                                      0, 2, 1,
                                      3).contiguous()  #.permute(2, 0, 3, 1, 4) # B, NH, N, C/NH
        k = self.k(key).reshape(B, L,
                                self.num_heads, C // self.num_heads).permute(
                                    0, 2, 1,
                                    3).contiguous()  #.permute(2, 0, 3, 1, 4) # B, NH, L, C/NH
        attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale # B, NH, N, L

        attn = attn.permute(0, 2, 3, 1) # B, N, L, NH
        #print('attn',attn.shape,hw_lvl)
        wedge_curr = 0
        feats_l = []
        for i in range(len(hw_lvl)):
            wedge_next = wedge_curr + hw_lvl[i][0] * hw_lvl[i][1]
            feats_l.append(attn[:, :, wedge_curr:wedge_next, :])

            feats_l[i] = self.linear_l[i](feats_l[i]).permute(0, 1, 3, 2).reshape(
                -1, self.num_heads, *hw_lvl[i])  # B*N,NH,H_i,W_i

            feats_l[i] = F.interpolate(feats_l[i], size=hw_lvl[0],
                                        mode='bilinear').permute(0, 2, 3, 1).reshape(
                                            B, N, -1, self.num_heads) # B,N,H_0*W_0,NH
            wedge_curr = wedge_next

        new_feats = torch.cat(feats_l, -1)
        mask = self.linear(new_feats)

        return mask


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm
                 ):
        super().__init__()
        self.fp16_enabled = False
        self.head_norm1 = norm_layer(dim)
        self.query_attn = QueryAttention(dim,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              attn_drop=attn_drop,
                              proj_drop=drop)
                 
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.head_norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)


    @force_fp32(apply_to=('query', 'key', 'value'))
    def forward(self, query, key, value, hw_lvl=None):
        query = self.query_attn(query, key, value, hw_lvl=hw_lvl)
        query = query + self.drop_path(query)
        query = self.head_norm1(query)

        query = query + self.drop_path(self.mlp(query))
        query = self.head_norm2(query)

        return query,


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-53296self.num_heads956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (
        x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(
        shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    @force_fp32(apply_to=('x', ))
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


@HEADS.register_module()
class DualFormerHead(BaseDecodeHead):
    def __init__(self,
                 num_head=2,
                 num_layers=1,
                 **kwargs):
        super(DualFormerHead, self).__init__(
            input_transform='multiple_select', **kwargs)

        # some Default Falue
        self.fp16_enabled = False
        mlp_ratio = 4
        qkv_bias = True
        qk_scale = None
        drop_rate = 0
        attn_drop_rate = 0

        norm_layer = None
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = None
        act_layer = act_layer or nn.GELU
        depth_model = self.in_channels[0]
        block = Block(dim=depth_model,
                      num_heads=num_head,
                      mlp_ratio=mlp_ratio,
                      qkv_bias=qkv_bias,
                      qk_scale=qk_scale,
                      drop=drop_rate,
                      attn_drop=attn_drop_rate,
                      drop_path=0,
                      norm_layer=norm_layer,
                      act_layer=act_layer)
        self.blocks = _get_clones(block, num_layers)
        self.attnen = AttentionTail(depth_model,
                                    num_heads=num_head,
                                    qkv_bias=qkv_bias,
                                    qk_scale=qk_scale,
                                    attn_drop=attn_drop_rate,
                                    proj_drop=0,
                                    n_channels = len(self.in_channels))
        self.stuff_query = nn.Embedding(self.num_classes,
                                        self.in_channels[0])
        self.stuff_query_pos = nn.Embedding(self.num_classes,
                                        self.in_channels[0])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        if pos is None:
            return tensor
        else:
            return tensor + pos
        #return tensor if pos is None else tensor + pos
    
    def forward(self, inputs):
        cat_list = []
        batch_sz = inputs[0].shape[0]
        channel_sz = self.in_channels[0]
        
        # generate hw_lvl and streched input
        hw_lvl = []
        for i,data in enumerate(inputs):
            if i in self.in_index:
                hw_lvl.append(data.shape[-2:])
                cat_list.append(data.permute(0,2,3,1).reshape(batch_sz, -1, channel_sz))
        streched_inp = torch.cat(cat_list, dim=1)

        # get semantic query
        query = [self.stuff_query.weight.unsqueeze(0) for i in range(batch_sz)]
        query = torch.cat(query, dim=0)
        query_pos = [self.stuff_query_pos.weight.unsqueeze(0) for i in range(batch_sz)]
        query_pos = torch.cat(query_pos, dim=0)

        result = self.calculate(streched_inp, None, query, query_pos, hw_lvl)
        result =  result.reshape(batch_sz, -1, hw_lvl[0][0], hw_lvl[0][1])
        return result

    @force_fp32(apply_to=('memory', 'pos_memory', 'query_embed',
                          'mask_query', 'pos_query'))
    def calculate(self, memory, pos_memory, query_embed, pos_query, hw_lvl):

        key_value =  self.with_pos_embed(memory, pos_memory)
        for i, block in enumerate(self.blocks):
            query_embed = block(
                                self.with_pos_embed(query_embed, pos_query),
                                key_value, # key
                                key_value, # value
                                hw_lvl=hw_lvl)

        attn = self.attnen(self.with_pos_embed(query_embed, pos_query),
                           key_value,
                           hw_lvl=hw_lvl)
        return attn
