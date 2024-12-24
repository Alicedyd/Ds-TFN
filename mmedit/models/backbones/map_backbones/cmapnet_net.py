import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from .mapnet_net import PriorDecodeLayer, SceneDecodeLayer

from .map_modules import ResidualBlocksWithInputConv
from .map_stda import STDABlock
from .map_utils import get_discrete_values, get_flow_from_grid, flow_warp_5d

from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, load_checkpoint
from mmedit.models import builder
from mmedit.models.registry import BACKBONES
from mmedit.models.common import PixelShufflePack
from mmedit.utils import get_root_logger


class ConfidenceBlock(nn.Module):
    def __init__(self,
                 channels,
                 level,):
        super().__init__()

        self.channels = channels
        self.level = level

        self.conv_layer = nn.Sequential(
                        nn.Conv2d(channels * 2, channels // 2, 3, 1, 1),
                        nn.LeakyReLU(negative_slope=0.1, inplace=True),
                    )

        self.confidence_head = nn.Sequential(
            nn.Conv2d(channels // 2, 1, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.reconstruction_head = nn.Conv2d(channels // 2, 3, 3, 1, 1)

    def forward(self, feats):
        # Get the pre_confidence
        confidence_feat = feats['confidence_feat'][-1]
        if self.level == 0:
            # The first confidence block
            feats['confidence'].append([])
            feats['pre_confidence'].append([])
            feats['pre_dehaze_result'].append([])
            pre_confidence = confidence_feat.contiguous()
        else:
            pre_confidence = feats['pre_confidence'][-1][self.level - 1]
        pre_confidence = torch.cat([pre_confidence, confidence_feat], dim=1)

        # Contact pre_confidence with former frame
        if len(feats['pre_confidence']) == 1:
            former_pre_confidence = pre_confidence.contiguous()
        else:
            former_pre_confidence_1 = feats['pre_confidence'][-2][self.level]
            former_pre_confidence_2 = feats['pre_confidence'][-2][self.level - 1]
            former_pre_confidence = torch.cat([former_pre_confidence_1, former_pre_confidence_2], dim=1)
        pre_confidence = torch.cat([pre_confidence, former_pre_confidence], dim=1)

        pre_confidence = self.conv_layer(pre_confidence)
        confidence = self.confidence_head(pre_confidence).add(1e-10)
        reconstruction_result = self.reconstruction_head(pre_confidence)

        feats['pre_confidence'][-1].append(pre_confidence)
        feats['confidence'][-1].append(confidence)
        feats['pre_dehaze_result'][-1].append(reconstruction_result)

        return feats


@BACKBONES.register_module()
class CMAPNet(BaseModule):
    """CMAP-Net.

    MAP-Net in "Video Dehazing via a Multi-Range Temporal Alignment Network with Physical Prior".
    Improved by adding confidence Module
    """
    RGB_MEAN = [0.485, 0.456, 0.406]
    RGB_STD = [0.229, 0.224, 0.225]

    def __init__(self,
                 backbone,
                 neck,
                 up_sampler,
                 confidence_num,
                 channels=32,
                 num_trans_bins=32,
                 align_depths=(1, 1, 1, 1),
                 num_kv_frames=(1, 2, 3),
                 ):
        super().__init__()

        self.backbone = builder.build_component(backbone)
        self.neck = builder.build_component(neck)
        self.up_sampler = builder.build_component(up_sampler)

        self.confidence_num = confidence_num
        self.confidence_upSampler = nn.UpsamplingNearest2d(scale_factor=4)
        self.confidence_layers = nn.ModuleList()
        for level in range(self.confidence_num):
            self.confidence_layers.append(
                ConfidenceBlock(
                    channels=64,
                    level=level,
                )
            )

        num_stages = len(align_depths)
        self.num_stages = num_stages

        # mpg
        self.num_trans_bins = num_trans_bins

        # msr: assume num_kv_frames is consecutive
        self.num_kv_frames = num_kv_frames

        # align & aggregate
        assert channels % 32 == 0
        num_heads = [channels // 32 for _ in range(num_stages)]
        kernel_sizes = [9, 7, 5, 3]

        self.prior_decoder_layers = nn.ModuleList()
        self.scene_decoder_layers = nn.ModuleList()

        guided_levels = (2, 3)  # memory consumption
        for s in range(num_stages):
            self.prior_decoder_layers.append(
                PriorDecodeLayer(
                    channels, s,
                    upsample=s < num_stages - 1, memory_enhance=s in guided_levels
                ))
            self.scene_decoder_layers.append(
                SceneDecodeLayer(
                    channels, s,
                    upsample=s < num_stages - 1, prior_guide=s in guided_levels,
                    num_kv_frames=num_kv_frames, align_depth=align_depths[s],
                    num_heads=num_heads[s], kernel_size=kernel_sizes[s]
                ))

        self.window_size = 32  # for padding
        rgb_mean = torch.Tensor(self.RGB_MEAN).reshape(1, 3, 1, 1)
        rgb_std = torch.Tensor(self.RGB_STD).reshape(1, 3, 1, 1)
        self.register_buffer('rgb_mean', rgb_mean)
        self.register_buffer('rgb_std', rgb_std)

    @property
    def with_neck(self):
        """bool: whether the segmentor has neck"""
        return hasattr(self, 'neck') and self.neck is not None

    def check_image_size(self, img):
        # https://github.com/JingyunLiang/SwinIR/blob/5aa89a7b275eeddc75cd7806378c89d23f298c48/main_test_swinir.py#L66
        # https://github.com/ZhendongWang6/Uformer/issues/32
        _, _, h, w = img.size()
        window_size = self.window_size
        mod_pad_h = (window_size - h % window_size) % window_size
        mod_pad_w = (window_size - w % window_size) % window_size
        out = F.pad(img, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return out

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def split_feat(self, feats, feat):
        feat_p, feat_j = [], []
        for s in range(self.num_stages):
            # Split the feats for prior and scene
            c = feat[s].shape[1]
            split_size_or_sections = c // 2
            x = torch.split(feat[s], split_size_or_sections, dim=1)
            feat_p.append(x[0])
            feat_j.append(x[1])

        feats['spatial_p'].append(feat_p)
        feats['spatial_j'].append(feat_j)

        feats['confidence_feat'].append(feat_p[0])
        return feats

    def confidence_calculate(self, feats):
        for i in range(self.confidence_num):
            feats = self.confidence_layers[i](feats)

        return feats

    def decode(self, feats):
        # init
        keys = ['decode_p', 'token_p', 'enhance_p',
                'decode_j', 'pos_j', 'ref_j']
        for k in keys:
            feats[k].append([None] * self.num_stages)
        keys = ['stage_t', 'stage_a', 'stage_j']
        for k in keys:
            feats[k] = [None] * self.num_stages

        for s in range(self.num_stages - 1, -1, -1):
            feats = self.prior_decoder_layers[s](feats)
            feats = self.scene_decoder_layers[s](feats)

        return feats

    def forward(self, lqs):
        """
        Forward function

        Args:
            lqs (Tensor): Input hazy sequence with shape (n, t, c, h, w).

        Returns:
            out (Tensor): Output haze-free sequence with shape (n, t, c, h, w).
        """
        n, T, c, h, w = lqs.shape

        feats = {
            'spatial_p': [], 'decode_p': [], 'token_p': [], 'enhance_p': [],
            'spatial_j': [], 'decode_j': [], 'pos_j': [], 'ref_j': [],
            'stage_j': [], 'stage_t': [], 'stage_a': [],

            'confidence': [], 'pre_confidence': [], 'confidence_feat': [], 'pre_dehaze_result': [],
        }

        out_js = []
        img_01s = []
        aux_js, aux_is = [], []

        for i in range(0, T):
            # print(f"\ntime: {i}")
            img = self.check_image_size(lqs[:, i, :, :, :])
            img_01 = img * self.rgb_std + self.rgb_mean  # to the range of [0., 1.]
            img_01s.append(img_01)

            # encode
            feat = self.extract_feat(img)  # tuple of feats, (4s, 8s, 16s, ...)
            feats = self.split_feat(feats, feat)
            feats = self.confidence_calculate(feats)

            # decode
            feats = self.decode(feats)

            # get output
            feat_j = feats['decode_j'][-1][0]
            confidence = feats['confidence'][-1][-1]
            _confidence = torch.ones(confidence.size(), device="cuda") - confidence

            feat_j = torch.mul(feat_j, _confidence) + torch.mul(feats['confidence_feat'][-1], confidence)
            out = self.up_sampler(feat_j)
            out = img_01 + out

            if self.training:
                assert h == out.shape[2] and w == out.shape[3]
            out_js.append(out[:, :, 0: h, 0: w].contiguous())

            # auxiliary output for the current timestep
            if self.training:
                aux_j, aux_i = [], []
                for s in range(self.num_stages):
                    tmp_j = F.interpolate(feats['stage_j'][s], size=img.shape[2:], mode='bilinear')
                    out_j = img_01 + tmp_j  # residue
                    tmp_t = F.interpolate(feats['stage_t'][s], size=img.shape[2:], mode='bilinear').clip(0, 1)
                    tmp_a = feats['stage_a'][s]
                    out_i = out_j * tmp_t + tmp_a * (1 - tmp_t)
                    aux_j.append(out_j[:, :, 0: h, 0: w])
                    aux_i.append(out_i[:, :, 0: h, 0: w])
                aux_js.append(aux_j)
                aux_is.append(aux_i)

            # memory management
            feats['spatial_j'].pop(0)
            feats['spatial_p'].pop(0)
            if len(feats['decode_j']) > max(self.num_kv_frames):
                feats['decode_j'].pop(0)
                feats['decode_p'].pop(0)
                feats['enhance_p'].pop(0)
                assert len(feats['decode_p']) == len(feats['decode_j'])
            if not self.training:
                feats['pos_j'].pop(0)
                feats['ref_j'].pop(0)

        out = dict(out=torch.stack(out_js, dim=1))  # output dict

        # for confidence training
        if self.training:
            confidence_list = [[] for i in range(self.confidence_num)]
            confidences = []

            pre_dehaze_list = [[] for i in range(self.confidence_num)]
            pre_dehaze = []

            for i in range(len(feats['pre_dehaze_result'])):
                for j in range(len(feats['pre_dehaze_result'][i])):
                    pre_dehaze_list[j].append(self.confidence_upSampler(feats['pre_dehaze_result'][i][j]) + img_01)
            for i in range(self.confidence_num):
                pre_dehaze.append(torch.stack(pre_dehaze_list[i], dim=1))

            for i in range(len(feats['confidence'])):
                for j in range(len(feats['confidence'][i])):
                    confidence_list[j].append(self.confidence_upSampler(feats['confidence'][i][j]))
            for i in range(self.confidence_num):
                confidences.append(torch.stack(confidence_list[i], dim=1))

            out['confidence'] = confidences
            out['pre_dehaze'] = pre_dehaze

        # auxiliary output for a sequence
        if self.training:
            pos, ref = [], []  # sampling locations
            for s in range(self.num_stages):
                pos.append(torch.stack([feats['pos_j'][i][s] for i in range(T)], dim=1))
                ref.append(torch.stack([feats['ref_j'][i][s] for i in range(T)], dim=1))
            out['pos'] = pos  # b, T, nr, g, h, w, 3
            out['ref'] = ref  # b, T, 1, h, w, 3
        if self.training:
            aux_j, aux_i = [], []  # Js, Is
            for s in range(self.num_stages):
                aux_j.append(torch.stack([aux_js[i][s] for i in range(T)], dim=1))
                aux_i.append(torch.stack([aux_is[i][s] for i in range(T)], dim=1))
            out['aux_j'] = aux_j
            out['aux_i'] = aux_i
            out['img_01'] = torch.stack(img_01s, dim=1)

        return out

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults: None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        logger = get_root_logger()
        logger.info(f"Init weights: {pretrained}")
        if isinstance(pretrained, str):
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif self.backbone.init_cfg is not None:
            self.backbone.init_weights()
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')
