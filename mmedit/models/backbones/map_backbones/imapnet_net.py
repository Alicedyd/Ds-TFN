from mmcv.runner import BaseModule, load_checkpoint
from mmedit.models import builder
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger


@BACKBONES.register_module()
class IMAPNet(BaseModule):
    """
     IMAP-Net fot video dehazing

     :param physic_based_dehazer: Net for preliminary dehazing by physical model
     :param histogram_refinement: Net for doing histogram refinement
    """

    def __init__(self,
                 physic_based_dehazer,
                 histogram_refinement,
                 ):
        super().__init__()

        self.pb_dehazer = builder.build_backbone(physic_based_dehazer)
        self.hr = builder.build_backbone(histogram_refinement)

    def forward(self, lqs):
        # Preliminary dehazing
        output = self.pb_dehazer(lqs)

        # Histogram Refinement
        pbd_out = output.pop('out')
        hr_out = self.hr(pbd_out)

        output['pbd_out'] = pbd_out
        output['hr_out'] = hr_out

        return output

    def init_weights(self, pretrained=None, strict=None):
        logger = get_root_logger()
        logger.info(f"Init weights: {pretrained}")
        if isinstance(pretrained, str):
            load_checkpoint(self, pretrained, strict=strict, logger=logger)


