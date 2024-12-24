import torch
import torch.nn.functional as F
import einops

from ..registry import MODELS
from .basic_dehazer import BasicDehazer

from mmedit.models.backbones.map_backbones.map_utils import flow_warp_5d, resize
from mmedit.models.losses.pixelwise_loss import l1_loss


def flow_loss(grid, ref, img0, img1, level):
    """
    see map_utils get_flow_from_grid
    """
    b, T, h, w, p = grid.shape
    assert p == 3, "Implementation for space-time flow warping"
    sf = 1. / 2 ** (level + 2)

    flow = (grid - ref).reshape(b * T, h, w, p)
    flow[:, :, :, 0] *= h
    flow[:, :, :, 1] *= w
    d = img0.shape[2]
    flow[:, :, :, 2] *= d
    assert flow.requires_grad

    # downsample and warp
    img0_lr = einops.rearrange(img0, 'bT c d h w -> (bT d) c h w')
    img0_lr = F.interpolate(img0_lr, scale_factor=sf, mode='bicubic')
    img0_lr = einops.rearrange(img0_lr, '(bT d) c h w -> bT c d h w', d=d)
    img0_lr_warp = flow_warp_5d(img0_lr, flow.unsqueeze(1))
    img0_lr_warp = img0_lr_warp.squeeze(2)
    img1_lr = F.interpolate(img1, scale_factor=sf, mode='bicubic')

    return l1_loss(img0_lr_warp, img1_lr)


@MODELS.register_module()
class IMAP(BasicDehazer):
    """
    IMAP Model for video dehazing

    :param generator(dict): Config for generator structure.
    :param pixel_loss(dict): Config for pixel-wise loss.
    :param train_cfg(dict): Config for training.
    :param test_cfg(dict): Config for testing.
    :param pretrained(str): Path for pretrained Model.
    """

    def __init__(self,
                 generator,
                 pixel_loss,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):

        super().__init__(generator,
                         pixel_loss,
                         train_cfg,
                         test_cfg,
                         pretrained)

        num_kv_frames = generator.get('physic_based_dehazer').get('num_kv_frames', 1)
        self.num_kv_frames = sorted(num_kv_frames) if isinstance(num_kv_frames, (list, tuple)) else [num_kv_frames]

    @staticmethod
    def _get_output_from_dict(x):
        if isinstance(x, dict):
            return x['hr_out']
        return x

    def forward_train(self, lq, gt):
        """
        Function for training forward

        :param lq: The input image with shape (n, T, c, h, w)
        :param gt: The ground truth of the input image with shape (n, T, c, h, w)
        :return: output(Tensor): the output Tensor
        """
        assert lq.ndim == 5 and lq.shape[1] > 1, f'Video dehazing method should have input t > 1 but get: {lq.shape}'
        losses = dict()
        output = self.generator(lq)
        loss_name = None

        n, T, c, h, w = gt.size()

        if isinstance(output, dict):
            for key in output.keys():
                if key == 'pbd_out':
                    continue
                elif key == 'confidence':
                    continue
                elif key == 'pre_dehaze':
                    loss_l = 0
                    loss_c = 0
                    for i in range(n):
                        for j in range(T):
                            gt_single = gt[i][j]
                            for k in range(3):
                                confidence = output['confidence'][k][i][j]
                                pre_dehaze = output['pre_dehaze'][k][i][j]

                                loss_l += torch.norm(
                                    (torch.mul(confidence, pre_dehaze) - torch.mul(confidence, gt_single)), p=1)
                                loss_c += torch.log(confidence).sum()
                    loss_name = key
                    loss_key = (loss_l - 0.01 * loss_c) / (T * 3)
                    loss_key *= 0.00001
                elif key == 'hr_out':
                    loss_name = key
                    loss_key = self.pixel_loss(output[key], gt)
                elif key == 'img_01':
                    continue
                elif key in ('aux_i', 'aux_j'):
                    loss_name = key.replace('aux_', 'phy-')
                    loss_key, lambda_phy = 0., 0.2
                    num_stages = len(output[key])
                    gt_key = gt if key == 'aux_j' else output['img_01']
                    for s in range(num_stages):
                        loss_weight = lambda_phy / (2 ** s)
                        loss_key += loss_weight * self.pixel_loss(output[key][s], gt_key)
                elif key.startswith('pos'):
                    assert len(output[key]) <= 4, f"pos should less than or equal to 4 but get: {len(output[key])}"
                    loss_name = 'flow'
                    loss_key, lambda_flow = 0., 0.2
                    num_stages = len(output[key])
                    for s in range(num_stages):
                        assert output[key][s].shape[-1] == 3
                        loss_weight = lambda_flow / 2 ** s
                        b, T, c, h, w = gt.shape
                        num_groups = output[key][s].size(3)
                        for g in range(num_groups):
                            num_kv_frames = self.num_kv_frames
                            img0s = []
                            for step in range(max(num_kv_frames)):
                                indices = torch.clip(torch.arange(T) - (step + 1), 0).to(gt.device)
                                img0 = torch.index_select(gt, dim=1, index=indices)
                                img0 = img0.reshape(b * T, 3, h, w)
                                img0s.append(img0)
                            img0s = torch.stack(img0s, dim=2)
                            for r, kv_frame in enumerate(num_kv_frames):
                                grid = output[key][s][:, :, r, g, :, :, :].clone()
                                ref = output[key.replace('pos', 'ref')][s].clone()
                                assert not ref.requires_grad
                                img0 = img0s[:, :, :kv_frame].clone()
                                img1 = gt.clone().reshape(b * T, 3, h, w)
                                loss_key += loss_weight * flow_loss(grid, ref, img0, img1, s)
                elif key.startswith('ref'):
                    continue
                losses[f'loss_{loss_name}'] = loss_key
            output = self._get_output_from_dict(output)
        else:
            loss_pix = self.pixel_loss(output, gt)
            losses['loss_pix'] = loss_pix

        outputs = dict(
            losses=losses,
            num_samples=len(gt.data),
            results=dict(lq=lq.cpu(), gt=gt.cpu(), output=output.cpu())
        )
        return outputs

    def forward_test(self,
                     lq,
                     gt=None,
                     meta=None,
                     save_image=False,
                     save_path=None,
                     iteration=None):
        """
        Function for testing forward

        :param lq: The input image with shape (n, c, h, w)
        :param gt: The ground truth of the input image with shape (n, c, h, w)
        :param meta: meta information for helping store results
        :param save_image: Whether save the image
        :param save_path: Path to save the image
        :param iteration: Iteration for the saving image name
        """
        torch.cuda.empty_cache()
        with torch.no_grad():
            assert lq.ndim == 5, "The IMAP Model is for video dehazing"
            output = self._get_output_from_dict(self.generator(lq))
        if lq.shape != gt.shape:
            # for REVIDE
            if not self.log_shape_warning:
                print(f"[Shape mismatch] lq: {lq.shape}, gt: {gt.shape}")
                self.log_shape_warning = True
            assert lq.shape[-2] == gt.shape[-2] // 2 and lq.shape[-1] == gt.shape[-1] // 2
            assert lq.ndim == 5
            outputs = []
            for i in range(output.size(1)):
                outputs.append(
                    resize(input=output[:, i, :, :, :],
                           size=gt.shape[-2:],
                           mode='bilinear',
                           align_corners=False)
                )
            output = torch.stack(outputs, dim=1)
        if self.test_cfg is not None and self.test_cfg.get('metrics', None):
            assert gt is not None, 'evaluations with metric must have gt images'
            results = dict(eval_result=self.evaluate(output, gt))
        else:
            results = dict(lq=lq.cpu(), output=output.cpu())
            if gt is not None:
                results['gt'] = gt.cpu()

        # save image
        if save_image:
            self._save_image(output, meta, save_path, iteration)

        return results

    def forward_use(self, lq):
        torch.cuda.empty_cache()
        with torch.no_grad():
            assert lq.ndim == 5, "The IMAP Model is for video dehazing"
            output = self._get_output_from_dict(self.generator(lq))
