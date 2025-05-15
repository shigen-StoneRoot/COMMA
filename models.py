from mamba_ssm import Mamba
import torch
from torch import nn
import numpy as np
from typing import Union, Type, List, Tuple
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list, get_matching_pool_op
from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp
from monai.networks.blocks import MLPBlock as Mlp
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from einops import rearrange
import itertools
import torch.nn.functional as F
from torch.nn import LayerNorm
from baselines import UNetDecoder
import math
from torch import Tensor
from dataloading_new import get_center
from einops.layers.torch import Rearrange

class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)



class UNet_Enc(nn.Module):
    def __init__(self):
        super(UNet_Enc, self).__init__()
        self.stages = [
            nn.Sequential(nn.Conv3d(1, 32, 3, 1, padding=1),
                          nn.InstanceNorm3d(32, eps=1e-5, affine=True),
                          nn.LeakyReLU(inplace=True)),
            nn.Sequential(nn.Conv3d(32, 64, 3, 2, padding=1),
                          nn.InstanceNorm3d(64, eps=1e-5, affine=True),
                          nn.LeakyReLU(inplace=True)),
            nn.Sequential(nn.Conv3d(64, 128, 3, 2, padding=1),
                          nn.InstanceNorm3d(128, eps=1e-5, affine=True),
                          nn.LeakyReLU(inplace=True)),
            nn.Sequential(nn.Conv3d(128, 256, 3, 2, padding=1),
                          nn.InstanceNorm3d(256, eps=1e-5, affine=True),
                          nn.LeakyReLU(inplace=True)),
            nn.Sequential(nn.Conv3d(256, 320, 3, 2, padding=1),
                          nn.InstanceNorm3d(320, eps=1e-5, affine=True),
                          nn.LeakyReLU(inplace=True)),
            nn.Sequential(nn.Conv3d(320, 320, 3, 2, padding=1),
                          nn.InstanceNorm3d(320, eps=1e-5, affine=True),
                          nn.LeakyReLU(inplace=True))
        ]
        self.stages = nn.ModuleList(self.stages)

    def forward(self, x):
        skips = []
        for s in self.stages:
            x = s(x)
            skips.append(x)
        return skips

    def step_forward(self, x, step):
        return self.stages[step](x)


class UNet_Dec(nn.Module):
    def __init__(self):
        super( UNet_Dec, self).__init__()
        # self.transpconvs = [
        #     nn.Sequential(nn.ConvTranspose3d(320, 320, kernel_size=4, stride=2, padding=1),
        #                   nn.InstanceNorm3d(320, eps=1e-5, affine=True)),  # 3->6
        #     nn.Sequential(nn.ConvTranspose3d(320, 256, kernel_size=4, stride=2, padding=1),
        #                   nn.InstanceNorm3d(256, eps=1e-5, affine=True)),  # 6->12
        #     nn.Sequential(nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
        #                   nn.InstanceNorm3d(128, eps=1e-5, affine=True)),  # 12->24
        #     nn.Sequential(nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
        #                   nn.InstanceNorm3d(64, eps=1e-5, affine=True)),  # 24->48
        #     nn.Sequential(nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
        #                   nn.InstanceNorm3d(32, eps=1e-5, affine=True)),  # 48->96
        # ]
        self.transpconvs = [
            nn.Sequential(
                          nn.Conv3d(320, 320, kernel_size=3, padding=1),
                          nn.InstanceNorm3d(320, eps=1e-5, affine=True),
                          nn.Upsample(scale_factor=2, mode='trilinear'),
                          ),  # 3->6
            nn.Sequential(nn.Conv3d(320, 256, kernel_size=3, padding=1),
                          nn.InstanceNorm3d(256, eps=1e-5, affine=True),
                          nn.Upsample(scale_factor=2, mode='trilinear'),
                          ),  # 6->12
            nn.Sequential(nn.Conv3d(256, 128, kernel_size=3, padding=1),
                          nn.InstanceNorm3d(128, eps=1e-5, affine=True),
                          nn.Upsample(scale_factor=2, mode='trilinear'),
                          ),  # 12->24
            nn.Sequential(nn.Conv3d(128, 64, kernel_size=3, padding=1),
                          nn.InstanceNorm3d(64, eps=1e-5, affine=True),
                          nn.Upsample(scale_factor=2, mode='trilinear'),
                          ),  # 24->48
            nn.Sequential(nn.Conv3d(64, 32, kernel_size=3, padding=1),
                          nn.InstanceNorm3d(32, eps=1e-5, affine=True),
                          nn.Upsample(scale_factor=2, mode='trilinear'),
                          ),  # 48->96
        ]
        self.transpconvs = nn.ModuleList(self.transpconvs)

        self.stages = [
            nn.Sequential(nn.Conv3d(320 * 2, 320, kernel_size=3, padding=1),
                          nn.InstanceNorm3d(320, eps=1e-5, affine=True),
                          nn.LeakyReLU(inplace=True)),
            nn.Sequential(nn.Conv3d(256 * 2, 256, kernel_size=3, padding=1),
                          nn.InstanceNorm3d(256, eps=1e-5, affine=True),
                          nn.LeakyReLU(inplace=True)),
            nn.Sequential(nn.Conv3d(128 * 2, 128, kernel_size=3, padding=1),
                          nn.InstanceNorm3d(128, eps=1e-5, affine=True),
                          nn.LeakyReLU(inplace=True)),
            nn.Sequential(nn.Conv3d(64 * 2, 64, kernel_size=3, padding=1),
                          nn.InstanceNorm3d(64, eps=1e-5, affine=True),
                          nn.LeakyReLU(inplace=True)),
            nn.Sequential(nn.Conv3d(32 * 2, 32, kernel_size=3, padding=1),
                          nn.InstanceNorm3d(32, eps=1e-5, affine=True),
                          nn.LeakyReLU(inplace=True)),
        ]
        self.stages = nn.ModuleList(self.stages)

        self.seg_layers = [
                           nn.Conv3d(320, 2, kernel_size=1, bias=True),
                           nn.Conv3d(256, 2, kernel_size=1, bias=True),
                           nn.Conv3d(128, 2, kernel_size=1, bias=True),
                           nn.Conv3d(64, 2, kernel_size=1, bias=True),
                           nn.Conv3d(32, 2, kernel_size=1, bias=True)]
        self.seg_layers = nn.ModuleList(self.seg_layers)

        self.deep_supervision = True

    def deep_supervision(self, deep_supervision):
        self.deep_supervision = deep_supervision

    def forward(self, skips):
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s+2)]), 1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r

    def step_forward(self, x, skip_connection, step):

        if torch.isnan(x).any():
            print('position 5')
            exit()
        x = self.transpconvs[step](x)

        if torch.isnan(x).any():
            print('position 6')
            print(step)
            exit()
        x = torch.cat((x, skip_connection), 1)
        x = self.stages[step](x)

        if torch.isnan(x).any():
            print('position 7')
            exit()

        if self.deep_supervision:
            ds_seg = self.seg_layers[step](x)
        elif step == (len(self.stages) - 1):
            ds_seg = self.seg_layers[-1](x)
        else:
            ds_seg = None
        return x, ds_seg


class PlainConvEncoder(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 return_skips: bool = False,
                 nonlin_first: bool = False,
                 pool: str = 'conv'
                 ):

        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        assert len(kernel_sizes) == n_stages, "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(features_per_stage) == n_stages, "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, "strides must have as many entries as we have resolution stages (n_stages). " \
                                             "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"

        stages = []
        for s in range(n_stages):
            stage_modules = []
            if pool == 'max' or pool == 'avg':
                if (isinstance(strides[s], int) and strides[s] != 1) or \
                        isinstance(strides[s], (tuple, list)) and any([i != 1 for i in strides[s]]):
                    stage_modules.append(get_matching_pool_op(conv_op, pool_type=pool)(kernel_size=strides[s], stride=strides[s]))
                conv_stride = 1
            elif pool == 'conv':
                conv_stride = strides[s]
            else:
                raise RuntimeError()
            stage_modules.append(StackedConvBlocks(
                n_conv_per_stage[s], conv_op, input_channels, features_per_stage[s], kernel_sizes[s], conv_stride,
                conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first
            ))
            stages.append(nn.Sequential(*stage_modules))
            input_channels = features_per_stage[s]

        self.stages = nn.Sequential(*stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        # we store some things that a potential decoder needs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x):
        ret = []
        for s in self.stages:
            x = s(x)
            ret.append(x)
        if self.return_skips:
            return ret
        else:
            return ret[-1]

    def step_forward(self, x, step):
        return self.stages[step](x)


class MambaLayer(nn.Module):
    def __init__(self, dim, in_dim, d_state=16, d_conv=4, expand=2, kernel_size=[3, 3, 3], stride=[1, 1, 1]):
        super().__init__()
        self.dim = dim
        self.dim_reduction = nn.Linear(in_dim, dim, bias=False)
        self.dim_expansion = nn.Linear(dim, in_dim, bias=False)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )

        self.mlp = Mlp(hidden_size=dim, mlp_dim=dim * 4, act="GELU", dropout_rate=0.0)

        # self.conv = StackedConvBlocks(1, nn.Conv3d, dim, dim, kernel_size, stride,
        #                               conv_bias=True, norm_op=nn.InstanceNorm3d,
        #                               norm_op_kwargs={'eps': 1e-5, 'affine': True},
        #                               dropout_op=None, dropout_op_kwargs=None, nonlin=nn.LeakyReLU,
        #                               nonlin_kwargs={'inplace': True}, nonlin_first=False)

    def forward(self, x_tokens):
        B, n_tokens, C = x_tokens.shape
        # assert C == self.dim

        x_flat = self.dim_reduction(x_tokens)
        x_residual = x_flat
        x_norm1 = self.norm1(x_flat)
        x_mamba = self.mamba(x_norm1)
        x_norm2 = self.norm2(x_mamba)
        mamba_out = x_norm2 + x_residual

        out = self.mlp(mamba_out) + mamba_out
        out = self.dim_expansion(out) + x_tokens
        return out


class BasicMambaLayer(nn.Module):
    def __init__(self, n_layer, dim, d_state=16, d_conv=4, expand=2, kernel_size=[3, 3, 3], stride=[1, 1, 1],
                 do_downsample=False, max_dim=None, token_size=(4, 4, 4), norm=True):
        super().__init__()
        self.token_size = token_size
        self.stack_mamba = [MambaLayer(dim, in_dim=dim * token_size[0] ** 3, d_state=d_state, d_conv=d_conv,
                                       expand=expand,
                                       kernel_size=kernel_size, stride=stride) for _ in range(n_layer)]
        self.stack_mamba = nn.ModuleList(self.stack_mamba)
        self.do_downsample = do_downsample
        self.do_norm = norm  # debug 是否加这个norm？
        if self.do_norm:
            self.norm = nn.InstanceNorm3d(dim)
        # if self.do_downsample:
        #     self.downsample = PatchMerging(dim, tar_dim=max_dim)

    def forward(self, x, token_size=None, pos_emb_layer=None):
        # B C H W D
        # if token_size is None:
        #     token_size = (1, 1, 1)
        token_size = self.token_size
        h, w, d = x.shape[2:]
        p1, p2, p3 = token_size
        img_shp = x.shape[2:]
        x0 = rearrange(x, 'b c (h p1) (w p2) (d p3) -> b (h w d) (c p1 p2 p3)', p1=p1, p2=p2, p3=p3)
        x1 = rearrange(x, 'b c (h p1) (w p2) (d p3) -> b (w d h) (c p2 p3 p1)', p1=p1, p2=p2, p3=p3)
        x2 = rearrange(x, 'b c (h p1) (w p2) (d p3) -> b (d h w) (c p3 p1 p2)', p1=p1, p2=p2, p3=p3)

        if pos_emb_layer is not None:
            indices_hwd = torch.arange(x0.shape[1]).view(h // p1, w // p2, d // p3).to(x.device)
            indices_wdh = indices_hwd.permute(1, 2, 0).contiguous().view(1, -1).to(x.device)
            indices_dhw = indices_hwd.permute(2, 0, 1).contiguous().view(1, -1).to(x.device)
            indices_hwd = indices_hwd.view(1, -1).to(x.device)

            x0 = x0 + pos_emb_layer(indices_hwd)
            x1 = x1 + pos_emb_layer(indices_wdh)
            x2 = x2 + pos_emb_layer(indices_dhw)

        for layer in self.stack_mamba:
            x = layer(x0) + layer(x1) + layer(x2)
        H, W, D = img_shp
        if self.do_downsample:
            x = rearrange(x, 'b (h w d) (c p1 p2 p3)-> b (h p1) (w p2) (d p3) c',
                          h=H // p1, w=W // p2, d=D // p3, p1=p1, p2=p2, p3=p3)
            x = self.downsample(x)  # B h/2 w/2 d/2 2c
            x = rearrange(x, 'b h w d c-> b c h w d')  # B (h/2 w/2 d/2) 2c
        else:

            x = rearrange(x, 'b (h w d) (c p1 p2 p3)-> b c (h p1) (w p2) (d p3)',
                          h=H // p1, w=W // p2, d=D // p3, p1=p1, p2=p2, p3=p3)
        if self.do_norm:
            x = self.norm(x)
        return x


# class depthwise_separable_conv(nn.Module):
#     def __init__(self, nin, kernels_per_layer, nout, stride):
#         super(depthwise_separable_conv, self).__init__()
#         self.depthwise = nn.Conv3d(nin, nin * kernels_per_layer, kernel_size=3, padding=1, groups=nin, stride=stride)
#         self.pointwise = nn.Conv3d(nin * kernels_per_layer, nout, kernel_size=1)
#
#     def forward(self, x):
#         out = self.depthwise(x)
#         out = self.pointwise(out)
#         return out

class MambaEncoder(nn.Module):
    def __init__(self, n_stages, layers_per_stage, dims_per_stage, d_state=16,
                 d_conv=4, expand=2, kernel_size=[3, 3, 3], stride=[1, 2, 2], do_downsample=True):
        super().__init__()
        # self.stages = nn.Sequential(nn.Sequential(
        #                                           depthwise_separable_conv(1, 3, dims_per_stage[0] // 2, stride=(1, 2, 2)),
        #                                           nn.InstanceNorm3d(dims_per_stage[0] // 2, eps=1e-5, affine=True),
        #                                           nn.LeakyReLU(inplace=True),
        #                                           depthwise_separable_conv(dims_per_stage[0] // 2, 1, dims_per_stage[0], stride=1),
        #                                           nn.InstanceNorm3d(dims_per_stage[0], eps=1e-5, affine=True),
        #                                           nn.LeakyReLU(inplace=True)
        #                                           ),
        #                             BasicMambaLayer(n_layer=layers_per_stage[0], dim=dims_per_stage[0],
        #                                             d_state=d_state, d_conv=d_conv, expand=expand,
        #                                             do_downsample=do_downsample),
        #                             BasicMambaLayer(n_layer=layers_per_stage[1], dim=dims_per_stage[1],
        #                                             d_state=d_state, d_conv=d_conv, expand=expand,
        #                                             do_downsample=do_downsample),
        #                             BasicMambaLayer(n_layer=layers_per_stage[2], dim=dims_per_stage[2],
        #                                             d_state=d_state, d_conv=d_conv, expand=expand,
        #                                             do_downsample=do_downsample),
        #                             BasicMambaLayer(n_layer=layers_per_stage[3], dim=dims_per_stage[3],
        #                                             d_state=d_state, d_conv=d_conv, expand=expand, max_dim=320,
        #                                             do_downsample=do_downsample),
        #                             )
        self.stages = nn.Sequential(nn.Sequential(nn.Conv3d(1, dims_per_stage[0] // 2, 3, stride=(1, 2, 2), padding=1),
                                                  nn.InstanceNorm3d(dims_per_stage[0] // 2, eps=1e-5, affine=True),
                                                  nn.LeakyReLU(inplace=True),
                                                  nn.Conv3d(dims_per_stage[0] // 2, dims_per_stage[0], 3,
                                                            stride=1, padding=1),
                                                  nn.InstanceNorm3d(dims_per_stage[0], eps=1e-5, affine=True),
                                                  nn.LeakyReLU(inplace=True)
                                                  ),
                                    BasicMambaLayer(n_layer=layers_per_stage[0], dim=dims_per_stage[0],
                                                    d_state=d_state, d_conv=d_conv, expand=expand,
                                                    do_downsample=do_downsample),
                                    BasicMambaLayer(n_layer=layers_per_stage[1], dim=dims_per_stage[1],
                                                    d_state=d_state, d_conv=d_conv, expand=expand,
                                                    do_downsample=do_downsample),
                                    BasicMambaLayer(n_layer=layers_per_stage[2], dim=dims_per_stage[2],
                                                    d_state=d_state, d_conv=d_conv, expand=expand,
                                                    do_downsample=do_downsample),
                                    BasicMambaLayer(n_layer=layers_per_stage[3], dim=dims_per_stage[3],
                                                    d_state=d_state, d_conv=d_conv, expand=expand, max_dim=320,
                                                    do_downsample=do_downsample),
                                    )
        self.pos_emb = nn.Embedding(32 * 32 * 24, dims_per_stage[0] * 4 ** 3)
    def forward(self, x):
        ret = []
        for s in self.stages:
            x = s(x)
            ret.append(x)
            return ret

    def step_forward(self, x, step, token_size=None):
        if step == 0:
            return self.stages[step](x)

        elif step == 1:
            return self.stages[step](x, token_size, self.pos_emb)
        else:
            return self.stages[step](x, token_size)

    def tokens2img(self, x_tokens, img_shp):
        h, w, d = img_shp
        x_img = rearrange(x_tokens, 'b (h w d) c-> b c h w d', h=h, w=w, d=d)
        return x_img


class Decoder(nn.Module):
    def __init__(self,
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision, nonlin_first: bool = False):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.num_classes = num_classes
        output_channels = [min(32 * 2 ** i, 320) for i in range(6)]
        strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
        conv_bias = True
        conv_op = nn.Conv3d
        kernel_sizes = [[3, 3, 3],  [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
        # kernel_sizes = [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
        norm_op = nn.InstanceNorm3d
        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op = None
        dropout_op_kwargs = None
        nonlin = nn.LeakyReLU
        nonlin_kwargs = {'inplace': True}

        n_stages_encoder = len(output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                          "resolution stages - 1 (n_stages in encoder - 1), " \
                                                          "here: %d" % n_stages_encoder

        transpconv_op = get_matching_convtransp(conv_op=nn.Conv3d)

        # we start with the bottleneck and work out way up
        stages = []
        transpconvs = []
        seg_layers = []
        for s in range(1, n_stages_encoder):
            input_features_below = output_channels[-s]
            input_features_skip = output_channels[-(s + 1)]
            stride_for_transpconv = strides[-s]
            transpconvs.append(transpconv_op(
                input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                bias=conv_bias
            ))
            # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
            stages.append(StackedConvBlocks(
                n_conv_per_stage[s-1], conv_op, 2 * input_features_skip, input_features_skip,
                kernel_sizes[-(s + 1)], 1, conv_bias, norm_op, norm_op_kwargs,
                dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first
            ))

            # we always build the deep supervision outputs so that we can always load parameters. If we don't do this
            # then a model trained with deep_supervision=True could not easily be loaded at inference time where
            # deep supervision is not needed. It's just a convenience thing
            seg_layers.append(conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s+2)]), 1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r

    def step_forward(self, x, skip_connection, step):

        x = self.transpconvs[step](x)
        x = torch.cat((x, skip_connection), 1)
        x = self.stages[step](x)
        if self.deep_supervision:
            ds_seg = self.seg_layers[step](x)
        elif step == (len(self.stages) - 1):
            ds_seg = self.seg_layers[-1](x)
        else:
            ds_seg = None
        return x, ds_seg

class CrossAttention(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        in_dim_q: int,
        in_dim_k: int,
        hidden_size: int,
        mlp_dim: int,
        num_heads: int,
        downsample_rate: int,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.dim_reduction_q = nn.Linear(in_dim_q, hidden_size, bias=False)
        self.dim_reduction_k = nn.Linear(in_dim_k, hidden_size, bias=False)
        self.dim_expansion = nn.Linear(hidden_size, in_dim_q, bias=False)
        self.dim_expansion_k = nn.Linear(hidden_size, in_dim_k, bias=False)
        self.mlp = Mlp(hidden_size, mlp_dim, dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = CrossAttention(hidden_size, num_heads, downsample_rate)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)

        self.pe = nn.Linear(3, hidden_size, bias=False)

    def forward(self, queries, keys, queries_coord=None, keys_coord=None):
        residual_keys = keys
        queries, keys = self.dim_reduction_q(queries), self.dim_reduction_k(keys)

        if queries_coord is not None:
            queries_pe, keys_pe = self.pe(queries_coord), self.pe(keys_coord)
            queries = queries + queries_pe
            keys = keys + keys_pe

        queries = self.norm1(queries)
        keys = self.norm3(keys)

        attn_out = self.attn(q=queries, k=keys, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        out = queries + mlp_out

        out = self.dim_expansion(out)
        return out, self.dim_expansion_k(keys) + residual_keys


# class PixelAttention(nn.Module):
#     def __init__(self, dim):
#         super(PixelAttention, self).__init__()
#         self.dim = dim
#         self.pa1 = nn.Conv3d(2, 1, 7, padding=3, bias=False)
#         self.pa2 = nn.Conv3d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x, pattn1):
#         pattn1 = self.pa1(pattn1).repeat((1, self.dim, 1, 1, 1))
#         B, C, H, W, D = x.shape
#         x = x.unsqueeze(dim=2) # B, C, 1, H, W, D
#         pattn1 = pattn1.unsqueeze(dim=2) # B, C, 1, H, W, D
#         x2 = torch.cat([x, pattn1], dim=2) # B, C, 2, H, W, D
#         x2 = Rearrange('b c t h w d -> b (c t) h w d')(x2)
#         pattn2 = self.pa2(x2)
#         pattn2 = self.sigmoid(pattn2)
#         return pattn2

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv3d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.concat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.ca = nn.Sequential(
            nn.Conv3d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn


class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv3d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W, D = x.shape
        x = x.unsqueeze(dim=2)  # B, C, 1, H, W, D
        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W, D
        x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W
        x2 = Rearrange('b c t h w d -> b (c t) h w d')(x2)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2


class CGAFusion(nn.Module):
    def __init__(self, dim, reduction=8):
        super(CGAFusion, self).__init__()
        self.dim_expansion = nn.Conv3d(32, dim, 1)
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv3d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_loc, x_glb):
        x_glb = self.dim_expansion(x_glb)
        initial = x_loc + x_glb
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = pattn2 * x_loc + (1 - pattn2) * x_glb
        result = self.conv(result) + x_loc
        return result


class COMMA(nn.Module):
    def __init__(self):
        super(COMMA, self).__init__()

        conv_kernel_sizes = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
        pool_op_kernel_sizes = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
        # n_conv_per_stage_encoder = [2, 2, 2, 2, 2, 2]
        n_conv_per_stage_encoder = [1, 1, 1, 1, 1, 1]
        n_stages = len(conv_kernel_sizes)
        self.n_stages = n_stages
        UNet_base_num_features = 32
        unet_max_num_features = 320
        mamba_channel = 32

        features_per_stage = [min(UNet_base_num_features * 2 ** i,
                                unet_max_num_features) for i in range(n_stages)]
        self.glb_encoder = MambaEncoder(5, layers_per_stage=[1, 1, 1, 1], dims_per_stage=[mamba_channel] * 4,
                                        do_downsample=False)

        self.loc_encoder = UNet_Enc()
        self.decoder = UNet_Dec()

        # cross attn pacth size
        # self.patch_sizes_L = [16, 8, 4, 2, 1, 1]

        # self.image_size = (96, 96, 96)
        # if self.image_size == (96, 96,96):
        #     self.patch_sizes_L = [1, 2, 3, 6] # patch size (96, 96, 96)
        # elif self.image_size == (64, 64, 64):
        #     self.patch_sizes_L = [1, 2, 2, 4]

        self.patch_sizes_L = [1, 2, 3, 6]  # patch size (96, 96, 96)
        self.patch_sizes_G = [8, 8, 8, 8]
        self.cross_hidden_sizes = [64] * 4 # [128, 128, 128, 256]
        # self.cross_hidden_sizes = [256, 256, 512, 512, 512, 1024]  # [64, 64, 128, 256, 512, 1024]
        self.mlp_dim = [size * 3 for size in self.cross_hidden_sizes]

        self.cross_attn = [CrossAttentionBlock(in_dim_q=features_per_stage[1:-1][::-1][i] * self.patch_sizes_L[i] ** 3,
                                               in_dim_k=mamba_channel * self.patch_sizes_G[i] ** 3,
                                               hidden_size=self.cross_hidden_sizes[i],
                                               mlp_dim=self.mlp_dim[i], num_heads=8, downsample_rate=2) # num_heads=8
                           for i in range(len(features_per_stage[1:-1]))]
        self.cross_attn = nn.ModuleList(self.cross_attn)

        # # loc-glb fusion
        features = [320, 256, 128, 64]
        # self.fusion_convs = [
        #     nn.Sequential(nn.Conv3d(features[i] + mamba_channel, features[i], kernel_size=1, padding=0),
        #                   nn.InstanceNorm3d(features[i], affine=True),
        #                   nn.LeakyReLU(inplace=True)
        #                   )
        #     for i in range(4)
        # ]
        # self.fusion_convs = nn.ModuleList(self.fusion_convs)

        # self.map_norm = nn.ModuleList([nn.InstanceNorm3d(features[i] + mamba_channel, affine=True) for i in range(4)])
        # self.fusion_maps = nn.ModuleList([PixelAttention(features[i] + mamba_channel) for i in range(4)])

        self.fusion_maps = nn.ModuleList([CGAFusion(features[i], reduction=8) for i in range(4)])

        self.cross_self = [
            nn.Sequential(nn.Conv3d(features[i] * 2, features[i], kernel_size=3, padding=1),
                          nn.InstanceNorm3d(features[i], affine=True),
                          nn.LeakyReLU(inplace=True))
            for i in range(4)
        ]
        self.cross_self = nn.ModuleList(self.cross_self)

        self.glb_prediction = nn.Conv3d(mamba_channel, 2, kernel_size=1, bias=False)

        # self.token_sizes = [None, (2, 2, 2)]



    def set_deep_supervision(self, deep_supervision=True):
        self.decoder.deep_supervision = deep_supervision

    def set_flips_angles(self, filps, angles):
        self.flips = filps
        self.angles = angles

    def set_glb_coord(self, glb_coord):
        self.glb_coord = glb_coord


    def forward(self, x_glb, x_loc, center):
        skips = self.loc_encoder(x_loc)

        x_loc = skips[-1]
        seg_outputs = []

        for s in range(len(self.decoder.stages)):
            x_loc, ds_seg = self.decoder.step_forward(x_loc, skips[-(s+2)], s)
            seg_outputs.append(ds_seg)
            if s == len(self.decoder.stages) - 1:
                break
            else:
                # None
                x_glb = self.glb_encoder.step_forward(x_glb, s)
                h_L, w_L, d_L = x_loc.shape[2:]
                h_G, w_G, d_G = x_glb.shape[2:]

                glb_patch = self.determine_crop_patch(center, x_glb, (h_L, w_L, d_L), (h_G, w_G, d_G))
                # glb_patch = self.get_fusion_features(x_loc, glb_patch, s)

                glb_patch = self.fusion_maps[s](x_loc, glb_patch)

                p_L = self.patch_sizes_L[s]
                p_G = self.patch_sizes_G[s]

                loc_coord = self.determine_patch_coord(center, (h_L, w_L, d_L))
                x_loc_coord = rearrange(loc_coord, 'b c (h p1) (w p2) (d p3) p -> b (h w d) (c p1 p2 p3) p',
                                        p1=p_L, p2=p_L, p3=p_L).mean(2).to(x_loc.device)
                glb_coord = self.determine_patch_coord([np.array([0.0, 0.0, 0.0])] * len(center), (h_G, w_G, d_G),
                                                       glb=True)

                # glb_coord = torch.rand(glb_coord.shape) * 2 - 1  ## shuffle ablation

                x_glb_coord = rearrange(glb_coord, 'b c (h p1) (w p2) (d p3) p -> b (h w d) (c p1 p2 p3) p',
                                        p1=p_G, p2=p_G, p3=p_G).mean(2).to(x_loc.device)



                x_loc = rearrange(x_loc, 'b c (h p1) (w p2) (d p3) -> b (h w d) (c p1 p2 p3)', p1=p_L, p2=p_L, p3=p_L)
                x_glb = rearrange(x_glb, 'b c (h p1) (w p2) (d p3) -> b (h w d) (c p1 p2 p3)', p1=p_G, p2=p_G, p3=p_G)

                x_loc, x_glb = self.cross_attn[s](x_loc, x_glb, x_loc_coord, x_glb_coord)

                x_loc = rearrange(x_loc, ' b (h w d) (c p1 p2 p3) -> b c (h p1) (w p2) (d p3)',
                                  p1=p_L, p2=p_L, p3=p_L, h=h_L // p_L, w=w_L // p_L, d=d_L // p_L)
                x_glb = rearrange(x_glb, ' b (h w d) (c p1 p2 p3) -> b c (h p1) (w p2) (d p3)',
                                  p1=p_G, p2=p_G, p3=p_G, h=h_G // p_G, w=w_G // p_G, d=d_G // p_G)

                # x_loc = x_loc + glb_patch

                x_loc = self.cross_self[s](torch.cat([x_loc, glb_patch], 1)) # Debug 最后是相加还是cat？


        seg_outputs = seg_outputs[::-1]
        if self.decoder.deep_supervision:
            # r = (seg_outputs, self.glb_prediction(F.interpolate(x_glb, size=(96, 128, 128), mode='trilinear')))
            r = (seg_outputs, self.glb_prediction(x_glb))
        else:
            r = seg_outputs[0]
        return r

    def forward_feat(self, x_glb, x_loc, center):
        skips = self.loc_encoder(x_loc)

        x_loc = skips[-1]
        seg_outputs = []

        for s in range(len(self.decoder.stages)):
            x_loc, ds_seg = self.decoder.step_forward(x_loc, skips[-(s+2)], s)
            seg_outputs.append(ds_seg)
            if s == len(self.decoder.stages) - 1:
                return x_loc
            else:
                # None
                x_glb = self.glb_encoder.step_forward(x_glb, s)
                h_L, w_L, d_L = x_loc.shape[2:]
                h_G, w_G, d_G = x_glb.shape[2:]

                glb_patch = self.determine_crop_patch(center, x_glb, (h_L, w_L, d_L), (h_G, w_G, d_G))
                # glb_patch = self.get_fusion_features(x_loc, glb_patch, s)

                glb_patch = self.fusion_maps[s](x_loc, glb_patch)

                p_L = self.patch_sizes_L[s]
                p_G = self.patch_sizes_G[s]

                loc_coord = self.determine_patch_coord(center, (h_L, w_L, d_L))
                x_loc_coord = rearrange(loc_coord, 'b c (h p1) (w p2) (d p3) p -> b (h w d) (c p1 p2 p3) p',
                                        p1=p_L, p2=p_L, p3=p_L).mean(2).to(x_loc.device)
                glb_coord = self.determine_patch_coord([np.array([0.0, 0.0, 0.0])] * len(center), (h_G, w_G, d_G),
                                                       glb=True)

                # glb_coord = torch.rand(glb_coord.shape) * 2 - 1  ## shuffle ablation

                x_glb_coord = rearrange(glb_coord, 'b c (h p1) (w p2) (d p3) p -> b (h w d) (c p1 p2 p3) p',
                                        p1=p_G, p2=p_G, p3=p_G).mean(2).to(x_loc.device)



                x_loc = rearrange(x_loc, 'b c (h p1) (w p2) (d p3) -> b (h w d) (c p1 p2 p3)', p1=p_L, p2=p_L, p3=p_L)
                x_glb = rearrange(x_glb, 'b c (h p1) (w p2) (d p3) -> b (h w d) (c p1 p2 p3)', p1=p_G, p2=p_G, p3=p_G)

                x_loc, x_glb = self.cross_attn[s](x_loc, x_glb, x_loc_coord, x_glb_coord)

                x_loc = rearrange(x_loc, ' b (h w d) (c p1 p2 p3) -> b c (h p1) (w p2) (d p3)',
                                  p1=p_L, p2=p_L, p3=p_L, h=h_L // p_L, w=w_L // p_L, d=d_L // p_L)
                x_glb = rearrange(x_glb, ' b (h w d) (c p1 p2 p3) -> b c (h p1) (w p2) (d p3)',
                                  p1=p_G, p2=p_G, p3=p_G, h=h_G // p_G, w=w_G // p_G, d=d_G // p_G)

                # x_loc = x_loc + glb_patch

                x_loc = self.cross_self[s](torch.cat([x_loc, glb_patch], 1)) # Debug 最后是相加还是cat？

    # def get_fusion_features(self, x_loc, glb_patch, s):
    #     res = torch.cat([x_loc, glb_patch], 1)
    #     res = self.map_norm[s](res)
    #     patten = torch.cat([torch.mean(res, dim=1, keepdim=True), torch.amax(res, dim=1, keepdim=True)], 1)
    #     patten = self.fusion_maps[s](res, patten)
    #     res = patten * res
    #     glb_patch = self.fusion_convs[s](res) + x_loc
    #     # glb_patch = res + x_loc
    #     return glb_patch

    def determine_crop_patch(self, center, x_glb, loc_shp, glb_shp):

        abs_centers = [get_center(glb_shp, [coor], flips, None) for coor, flips in zip(center, self.flips)]
        h_L, w_L, d_L = loc_shp
        h_G, w_G, d_G = glb_shp
        ph_G, pw_G, pd_G = h_L // 2, w_L // 2, d_L // 2

        crop_patch = []
        for idx, abs_center in enumerate(abs_centers):

            X_min, X_max = max(abs_center[0] - ph_G, 0), min(abs_center[0] + ph_G, h_G)
            Y_min, Y_max = max(abs_center[1] - pw_G, 0), min(abs_center[1] + pw_G, w_G)
            Z_min, Z_max = max(abs_center[2] - pd_G, 0), min(abs_center[2] + pd_G, d_G)

            glb_patch = x_glb[idx, :, X_min: X_max, Y_min: Y_max, Z_min: Z_max].unsqueeze(0)
            if glb_patch.shape[1:] != loc_shp:
                glb_patch = F.interpolate(glb_patch, mode='trilinear', size=(h_L, w_L, d_L))

            crop_patch.append(glb_patch)

        crop_patch = torch.cat(crop_patch, 0)
        return crop_patch

    def determine_patch_coord(self, center_coordinates, shp, glb=False):
        assert self.glb_coord is not None
        depth, height, width = shp

        relative_coordinates = []
        # 计算每个像素点的相对坐标
        for idx, center_coordinate in enumerate(center_coordinates):
            if glb:
                max_x, min_x = 1, -1
                max_y, min_y = 1, -1
                max_z, min_z = 1, -1
            else:
                max_x, min_x = center_coordinate[0] + 48 / self.glb_coord[idx][0] / 2, \
                               center_coordinate[0] - 48 / self.glb_coord[idx][0] / 2

                max_y, min_y = center_coordinate[1] + 64 / self.glb_coord[idx][1] / 2, \
                               center_coordinate[1] - 64 / self.glb_coord[idx][1] / 2

                max_z, min_z = center_coordinate[2] + 64 / self.glb_coord[idx][2] / 2, \
                               center_coordinate[2] - 64 / self.glb_coord[idx][2] / 2

            grid_d, grid_h, grid_w = np.meshgrid(np.linspace(max(-1, min_x), min(1, max_x), depth),
                                                 np.linspace(max(-1, min_y), min(1, max_y), height),
                                                 np.linspace(max(-1, min_z), min(1, max_z), width),
                                                 indexing='ij')

            relative_coordinate = np.stack((grid_d,
                                            grid_h,
                                            grid_w), axis=-1)
            relative_coordinates.append(relative_coordinate[np.newaxis, np.newaxis, :])

        return torch.from_numpy(np.concatenate(relative_coordinates, 0)).float()

class SLAMamba3(nn.Module):
    def __init__(self):
        super(SLAMamba3, self).__init__()

        conv_kernel_sizes = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
        pool_op_kernel_sizes = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
        # n_conv_per_stage_encoder = [2, 2, 2, 2, 2, 2]
        n_conv_per_stage_encoder = [1, 1, 1, 1, 1, 1]
        n_stages = len(conv_kernel_sizes)
        self.n_stages = n_stages
        UNet_base_num_features = 32
        unet_max_num_features = 320
        mamba_channel = 32

        features_per_stage = [min(UNet_base_num_features * 2 ** i,
                                  unet_max_num_features) for i in range(n_stages)]
        self.glb_encoder = MambaEncoder(5, layers_per_stage=[1, 1, 1, 1], dims_per_stage=[mamba_channel] * 4,
                                        do_downsample=False)

        self.loc_encoder = UNet_Enc()
        self.decoder = UNet_Dec()

        # cross attn pacth size
        # self.patch_sizes_L = [16, 8, 4, 2, 1, 1]

        # self.image_size = (96, 96, 96)
        # if self.image_size == (96, 96,96):
        #     self.patch_sizes_L = [1, 2, 3, 6] # patch size (96, 96, 96)
        # elif self.image_size == (64, 64, 64):
        #     self.patch_sizes_L = [1, 2, 2, 4]

        self.patch_sizes_L = [1, 2, 3, 6]  # patch size (96, 96, 96)
        self.patch_sizes_G = [8, 8, 8, 8]
        self.cross_hidden_sizes = [64] * 4  # [128, 128, 128, 256]
        # self.cross_hidden_sizes = [256, 256, 512, 512, 512, 1024]  # [64, 64, 128, 256, 512, 1024]
        self.mlp_dim = [size * 3 for size in self.cross_hidden_sizes]

        self.cross_attn = [
            CrossAttentionBlock(in_dim_q=features_per_stage[1:-1][::-1][i] * self.patch_sizes_L[i] ** 3,
                                in_dim_k=mamba_channel * self.patch_sizes_G[i] ** 3,
                                hidden_size=self.cross_hidden_sizes[i],
                                mlp_dim=self.mlp_dim[i], num_heads=8, downsample_rate=2)  # num_heads=8
            for i in range(len(features_per_stage[1:-1]))]
        self.cross_attn = nn.ModuleList(self.cross_attn)

        # # loc-glb fusion
        features = [320, 256, 128, 64]
        # self.fusion_convs = [
        #     nn.Sequential(nn.Conv3d(features[i] + mamba_channel, features[i], kernel_size=1, padding=0),
        #                   nn.InstanceNorm3d(features[i], affine=True),
        #                   nn.LeakyReLU(inplace=True)
        #                   )
        #     for i in range(4)
        # ]
        # self.fusion_convs = nn.ModuleList(self.fusion_convs)

        # self.map_norm = nn.ModuleList([nn.InstanceNorm3d(features[i] + mamba_channel, affine=True) for i in range(4)])
        # self.fusion_maps = nn.ModuleList([PixelAttention(features[i] + mamba_channel) for i in range(4)])

        self.fusion_maps = nn.ModuleList([CGAFusion(features[i], reduction=8) for i in range(4)])

        self.cross_self = [
            nn.Sequential(nn.Conv3d(features[i] * 2, features[i], kernel_size=3, padding=1),
                          nn.InstanceNorm3d(features[i], affine=True),
                          nn.LeakyReLU(inplace=True))
            for i in range(4)
        ]
        self.cross_self = nn.ModuleList(self.cross_self)

        self.glb_prediction = nn.Conv3d(mamba_channel, 2, kernel_size=1, bias=False)

        # self.token_sizes = [None, (2, 2, 2)]

    def set_deep_supervision(self, deep_supervision=True):
        self.decoder.deep_supervision = deep_supervision

    def set_flips_angles(self, filps, angles):
        self.flips = filps
        self.angles = angles

    def set_glb_coord(self, glb_coord):
        self.glb_coord = glb_coord

    def forward(self, x_glb, x_loc, center):
        skips = self.loc_encoder(x_loc)

        x_loc = skips[-1]
        seg_outputs = []

        for s in range(len(self.decoder.stages)):
            x_loc, ds_seg = self.decoder.step_forward(x_loc, skips[-(s + 2)], s)
            seg_outputs.append(ds_seg)
            if s == len(self.decoder.stages) - 1:
                break
            else:
                # None
                residual = x_loc
                x_glb = self.glb_encoder.step_forward(x_glb, s)
                h_L, w_L, d_L = x_loc.shape[2:]
                h_G, w_G, d_G = x_glb.shape[2:]

                glb_patch = self.determine_crop_patch(center, x_glb, (h_L, w_L, d_L), (h_G, w_G, d_G))


                glb_patch = self.fusion_maps[s](x_loc, glb_patch)

                p_L = self.patch_sizes_L[s]
                p_G = self.patch_sizes_G[s]

                loc_coord = self.determine_patch_coord(center, (h_L, w_L, d_L))
                x_loc_coord = rearrange(loc_coord, 'b c (h p1) (w p2) (d p3) p -> b (h w d) (c p1 p2 p3) p',
                                        p1=p_L, p2=p_L, p3=p_L).mean(2).to(x_loc.device)
                glb_coord = self.determine_patch_coord([np.array([0.0, 0.0, 0.0])] * len(center), (h_G, w_G, d_G),
                                                       glb=True)
                x_glb_coord = rearrange(glb_coord, 'b c (h p1) (w p2) (d p3) p -> b (h w d) (c p1 p2 p3) p',
                                        p1=p_G, p2=p_G, p3=p_G).mean(2).to(x_loc.device)

                x_loc = rearrange(x_loc, 'b c (h p1) (w p2) (d p3) -> b (h w d) (c p1 p2 p3)', p1=p_L, p2=p_L,
                                  p3=p_L)
                x_glb = rearrange(x_glb, 'b c (h p1) (w p2) (d p3) -> b (h w d) (c p1 p2 p3)', p1=p_G, p2=p_G,
                                  p3=p_G)

                x_loc, x_glb = self.cross_attn[s](x_loc, x_glb, x_loc_coord, x_glb_coord)

                x_loc = rearrange(x_loc, ' b (h w d) (c p1 p2 p3) -> b c (h p1) (w p2) (d p3)',
                                  p1=p_L, p2=p_L, p3=p_L, h=h_L // p_L, w=w_L // p_L, d=d_L // p_L)
                x_glb = rearrange(x_glb, ' b (h w d) (c p1 p2 p3) -> b c (h p1) (w p2) (d p3)',
                                  p1=p_G, p2=p_G, p3=p_G, h=h_G // p_G, w=w_G // p_G, d=d_G // p_G)

                # x_loc = x_loc + glb_patch

                x_loc = self.cross_self[s](torch.cat([x_loc, glb_patch], 1)) + residual # Debug 最后是相加还是cat？

        seg_outputs = seg_outputs[::-1]
        if self.decoder.deep_supervision:
            # r = (seg_outputs, self.glb_prediction(F.interpolate(x_glb, size=(96, 128, 128), mode='trilinear')))
            r = (seg_outputs, self.glb_prediction(x_glb))
        else:
            r = seg_outputs[0]
        return r

    # def get_fusion_features(self, x_loc, glb_patch, s):
    #     res = torch.cat([x_loc, glb_patch], 1)
    #     res = self.map_norm[s](res)
    #     patten = torch.cat([torch.mean(res, dim=1, keepdim=True), torch.amax(res, dim=1, keepdim=True)], 1)
    #     patten = self.fusion_maps[s](res, patten)
    #     res = patten * res
    #     glb_patch = self.fusion_convs[s](res) + x_loc
    #     # glb_patch = res + x_loc
    #     return glb_patch

    def determine_crop_patch(self, center, x_glb, loc_shp, glb_shp):

        abs_centers = [get_center(glb_shp, [coor], flips, None) for coor, flips in zip(center, self.flips)]
        h_L, w_L, d_L = loc_shp
        h_G, w_G, d_G = glb_shp
        ph_G, pw_G, pd_G = h_L // 2, w_L // 2, d_L // 2

        crop_patch = []
        for idx, abs_center in enumerate(abs_centers):

            X_min, X_max = max(abs_center[0] - ph_G, 0), min(abs_center[0] + ph_G, h_G)
            Y_min, Y_max = max(abs_center[1] - pw_G, 0), min(abs_center[1] + pw_G, w_G)
            Z_min, Z_max = max(abs_center[2] - pd_G, 0), min(abs_center[2] + pd_G, d_G)

            glb_patch = x_glb[idx, :, X_min: X_max, Y_min: Y_max, Z_min: Z_max].unsqueeze(0)
            if glb_patch.shape[1:] != loc_shp:
                glb_patch = F.interpolate(glb_patch, mode='trilinear', size=(h_L, w_L, d_L))

            crop_patch.append(glb_patch)

        crop_patch = torch.cat(crop_patch, 0)
        return crop_patch

    def determine_patch_coord(self, center_coordinates, shp, glb=False):
        assert self.glb_coord is not None
        depth, height, width = shp

        relative_coordinates = []
        # 计算每个像素点的相对坐标
        for idx, center_coordinate in enumerate(center_coordinates):
            if glb:
                max_x, min_x = 1, -1
                max_y, min_y = 1, -1
                max_z, min_z = 1, -1
            else:
                max_x, min_x = center_coordinate[0] + 48 / self.glb_coord[idx][0] / 2, \
                               center_coordinate[0] - 48 / self.glb_coord[idx][0] / 2

                max_y, min_y = center_coordinate[1] + 64 / self.glb_coord[idx][1] / 2, \
                               center_coordinate[1] - 64 / self.glb_coord[idx][1] / 2

                max_z, min_z = center_coordinate[2] + 64 / self.glb_coord[idx][2] / 2, \
                               center_coordinate[2] - 64 / self.glb_coord[idx][2] / 2

            grid_d, grid_h, grid_w = np.meshgrid(np.linspace(max(-1, min_x), min(1, max_x), depth),
                                                 np.linspace(max(-1, min_y), min(1, max_y), height),
                                                 np.linspace(max(-1, min_z), min(1, max_z), width),
                                                 indexing='ij')

            relative_coordinate = np.stack((grid_d,
                                            grid_h,
                                            grid_w), axis=-1)
            relative_coordinates.append(relative_coordinate[np.newaxis, np.newaxis, :])

        return torch.from_numpy(np.concatenate(relative_coordinates, 0)).float()

# def forward_encoder(self, x_glb, x_loc):
#     ret = []
#     for step in range(self.n_stages):
#         x_glb = self.glb_encoder.step_forward(x_glb, step)
#         x_loc = self.loc_encoder.step_forward(x_loc, step)
#         #
#         h, w, d = x_loc.shape[2:]
#
#         p = self.patch_sizes[step]
#         x_loc = rearrange(x_loc, 'b c (h p1) (w p2) (d p3) -> b (h w d) (c p1 p2 p3)', p1=p, p2=p, p3=p)
#         x_glb = rearrange(x_glb, 'b c (h p1) (w p2) (d p3) -> b (h w d) (c p1 p2 p3)', p1=p, p2=p, p3=p)
#
#         x_loc = self.cross_attn[step](x_loc, x_glb) + x_loc  # x_loc = self.cross_attn[step](x_loc, x_glb) + x_loc
#
#         x_loc = rearrange(x_loc, ' b (h w d) (c p1 p2 p3) -> b c (h p1) (w p2) (d p3)',
#                           p1=p, p2=p, p3=p, h=h // p, w=w // p, d=d // p)
#         x_glb = rearrange(x_glb, ' b (h w d) (c p1 p2 p3) -> b c (h p1) (w p2) (d p3)',
#                           p1=p, p2=p, p3=p, h=h // p, w=w // p, d=d // p)
#
#         ret.append(x_loc)
#     return ret
#
# def forward(self, x_glb, x_loc):
#     skips = self.forward_encoder(x_glb, x_loc)
#     return self.decoder(skips)



if __name__ == '__main__':
    device = torch.device('cuda:3')
    # model = BasicMambaLayer(2, 16).to(device)
    x_loc = torch.randn(2, 1, 96, 96, 96).to(device)
    x_glb = torch.randn(2, 1, 96, 192, 192).to(device)

    model = SLAMamba().to(device)

    # print(model)

    # model.set_flips_angles(np.array([0, 0, 0]), np.array([0, 0, 0]))
    # model.set_deep_supervision(False)
    # feats = model(x_glb, x_loc, [0.1, -0.1, 0.4])
    # for feat in [feats]:
    #     print(feat.shape)
    #
    # model = SLAMamba().to(device)
    # feats_loc = model.forward_loc(x_loc)
    # feats_glb = model.forward_glb(x_glb)
    #
    # for feat in feats_loc:
    #     print(feat.shape)
    #
    # for feat in feats_glb:
    #     print(feat.shape)

    # cross_attn = CrossAttentionBlock(in_dim=32 * 16 ** 3, hidden_size=64, mlp_dim=64*4, num_heads=8, downsample_rate=2).to(device)
    #
    # q = rearrange(x_loc, 'b c (h p1) (w p2) (d p3) -> b (h w d) (c p1 p2 p3)', p1=16, p2=16, p3=16)
    # k = rearrange(x_glb, 'b c (h p1) (w p2) (d p3) -> b (h w d) (c p1 p2 p3)', p1=16, p2=16, p3=16)
    #
    # out = cross_attn(q, k)
    # out = rearrange(out, ' b (h w d) (c p1 p2 p3) -> b c (h p1) (w p2) (d p3)', p1=16, p2=16, p3=16, h=6, w=6, d=6)
    # print(out.shape)
    # patch_emd = nn.Conv3d(16, 16, kernel_size=2, stride=2).to(device)
    # x_tokens = patch_emd(x)
    # x_tokens = rearrange(x_tokens, 'b c h w d -> b c (h w d)')
    # x_tokens = x_tokens.transpose(-1, -2)
    #
    # print(x_tokens.shape)
    # x_tokens = model(x_tokens, (96 // 2, 96 // 2, 96 // 2))
    # print(x_tokens.shape)
    #
    # x_out = rearrange(x_tokens, 'b (h w d) c-> b h w d c', h=24, w=24, d=24)
    # print(x_out.shape)
    #
    # downsample = PatchMerging(16).to(device)
    # print(downsample(x_out).shape)

    # n_stages = 6
    # base_num_features = 32
    # max_num_features = 320
    # features_per_stage = [min(base_num_features * 2 ** i,
    #                           max_num_features) for i in range(n_stages)]
    # pool_op_kernel_sizes = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]]
    # conv_kernel_sizes = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
    # n_conv_per_stage = [2, 2, 2, 2, 2, 2]
    # model = PlainConvEncoder(input_channels=1, n_stages=n_stages, features_per_stage=features_per_stage,
    #                          conv_op=nn.Conv3d, kernel_sizes=conv_kernel_sizes, strides=pool_op_kernel_sizes,
    #                          n_conv_per_stage=n_conv_per_stage, conv_bias=True, norm_op=nn.InstanceNorm3d,
    #                          norm_op_kwargs={'eps': 1e-5, 'affine': True}, dropout_op=None,
    #                          dropout_op_kwargs=None, nonlin=nn.LeakyReLU, nonlin_kwargs={'inplace': True},
    #                          return_skips=True, nonlin_first=False).to(device)
    #
    # feats = model(x)
    # for feat in feats:
    #     print(feat.shape)

    # conv = StackedConvBlocks(2, nn.Conv3d, 1, 4, [3, 3, 3], [1, 1, 1],
    #                          conv_bias=True, norm_op=nn.InstanceNorm3d,
    #                          norm_op_kwargs={'eps': 1e-5, 'affine': True},
    #                          dropout_op=None, dropout_op_kwargs=None, nonlin=nn.LeakyReLU,
    #                          nonlin_kwargs={'inplace': True}, nonlin_first=False).to(device)
    #
    # print(conv(x).shape)

    # model = MambaEncoder(5, layers_per_stage=[2, 2, 2, 2, 2], dims_per_stage=[32, 64, 128, 256, 320]).to(device)
    #
    # outs = model(x)
