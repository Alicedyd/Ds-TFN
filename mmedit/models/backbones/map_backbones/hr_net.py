import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from mmedit.models.registry import BACKBONES
from mmcv.runner import BaseModule, load_checkpoint
from mmedit.models import builder
from mmedit.utils import get_root_logger


class HistogramRefinementDecoder(nn.Module):
    """
    Decoder for histogram refinement with a FPN structure

    :param fusion(bool): Whether to make a fusion feature
    :param num_stages(int): The number of layers in featuren pyramid
    :param channels(int): The channels of input features
    :param num_trans_bins(int): The channels of output features
    """

    def __init__(self,
                 fusion=False,
                 num_stages=4,
                 channels=128,
                 num_trans_bins=128):
        super(HistogramRefinementDecoder, self).__init__()

        self.fusion = fusion
        self.num_stages = num_stages
        self.channels = channels
        self.num_trans_bins = num_trans_bins

        self.up_sample = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv_layers = nn.ModuleList()
        self.heads = nn.ModuleList()
        for i in range(num_stages):
            self.conv_layers.append(nn.Conv2d(channels, channels, 1, 1, 0))
            if self.fusion:
                self.heads.append(
                    nn.Sequential(
                        nn.Conv2d(channels, channels, 3, 1, 1),
                        nn.LeakyReLU(negative_slope=0.1, inplace=True),
                        nn.Conv2d(channels, num_trans_bins, 1, 1, 0)
                    )
                )

    def forward(self, feats):
        new_feats = []
        last_feat = None
        for i in range(len(feats)):
            _feat = feats[i].contiguous()

            _feat = self.conv_layers[i](_feat)
            if last_feat is not None:
                _feat = _feat + self.up_sample(last_feat)
            last_feat = _feat.contiguous()

            if self.fusion:
                # Use the head
                _feat = self.heads[i](_feat)
            else:
                _feat += feats[i]
            new_feats.append(_feat)

        if self.fusion:
            # A fusion feature
            fusion_feat = new_feats[0]
            for i in range(1, self.num_stages):
                _output = new_feats[i]
                fusion_feat = _output + self.up_sample(fusion_feat)
            return fusion_feat
        else:
            # All features
            return new_feats


class DehazeRefinementDecoder(nn.Module):
    """
    Decoder for dehaze refinement with a FPN structure

    :param num_stages(int): The number of layers in featuren pyramid
    :param channels(int): The channels of input features
    """
    def __init__(self,
                 channels=128,
                 num_trans_bins=128):
        super(DehazeRefinementDecoder, self).__init__()

        self.channels = channels
        self.num_trans_bins = num_trans_bins

        self.head = nn.Sequential(
                        nn.Conv2d(channels, channels, 3, 1, 1),
                        nn.LeakyReLU(negative_slope=0.1, inplace=True),
                        nn.Conv2d(channels, num_trans_bins, 1, 1, 0)
                    )
        self.conv1x1_layer = nn.Conv2d(channels, channels, 1, 1, 0)
        self.upSample = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, feat, feat_h):
        """
        feat: Feature from current layer
        feat_h: Feature from higher layer
        """

        if feat_h is not None:
            _feat = self.conv1x1_layer(feat) + self.upSample(feat_h)
        else:
            _feat = feat
        result = self.head(_feat)

        return result


@BACKBONES.register_module()
class HRNet(BaseModule):
    """
    Histogram Refinement Net in IMAP-Net for video dehazing

    :param backbone: Backbone of the Net
    :param neck: Neck of the Net
    :param up_sampler: Up-sampler of the Net
    :param num_stages: The number of feats used
    :param window_size: The atomic size of the img
    """

    def __init__(self,
                 backbone,
                 neck,
                 up_sampler,
                 num_stages,
                 window_size):
        super().__init__()

        self.backbone = builder.build_component(backbone)
        self.neck = builder.build_component(neck)
        self.up_sampler = builder.build_component(up_sampler)

        self.num_stages = num_stages
        self.window_size = window_size

        # Layers
        self.decoders = nn.ModuleList()
        for i in range(num_stages-1):
            self.decoders.append(
                DehazeRefinementDecoder()
            )
        self.decoders.append(
            DehazeRefinementDecoder(num_trans_bins=32)
        )

    @property
    def _with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    def check_image_size(self, img):
        _, _, h, w = img.size()
        window_size = self.window_size
        mod_pad_h = (window_size - h % window_size) % window_size
        mod_pad_w = (window_size - w % window_size) % window_size
        out = F.pad(img, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return out

    def extract_feat(self, img):
        out = self.backbone(img)
        if self._with_neck:
            out = self.neck(out)
        return out

    def decode(self, old_feat, feat):
        feats = []
        for i in range(self.num_stages):
            _old_feat = old_feat[i]
            _feat = feat[i]

            contact_feat = torch.cat([_feat, _old_feat], dim=1)
            feats.append(contact_feat)
        feats.reverse()

        feat_h = None
        for i in range(self.num_stages):
            feat_h = self.decoders[i](feats[i], feat_h)
        return feat_h

    def forward(self, lqs):
        n, T, c, h, w = lqs.shape
        old_feat = None

        hr_outs = []

        for i in range(T):
            img = self.check_image_size(lqs[:, i, :, :, :])  # get single image

            # encode
            feat = self.extract_feat(img)

            # decode
            if old_feat is None:
                old_feat = feat
            hr_out = self.decode(old_feat, feat)
            hr_out = img + self.up_sampler(hr_out)

            hr_outs.append(hr_out[:, :, 0: h, 0: w].contiguous())

            old_feat = feat

        return torch.stack(hr_outs, dim=1)

    def init_weights(self, pretrained=None, strict=None):
        """
        Initialize the weights of the Model

        :param pretrained: Pretrained Model weight path
        :param strict: Whether strictly load the pretrained model
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
