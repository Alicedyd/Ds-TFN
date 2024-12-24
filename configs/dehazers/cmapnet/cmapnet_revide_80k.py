_base_ = [
    '../_base_/datasets/revide.py',
    '../_base_/default_runtime.py',
    './cmapnet_runtime.py',
    '../_base_/schedules/schedule_80k_eval.py'
]

exp_name = 'cmapnet_revide_80k'

checkpoint = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-tiny_3rdparty_32xb128-noema_in1k_20220301-795e9634.pth'  # noqa

# model settings
CMAPNet = dict(
    type='CMAPNet',
    backbone=dict(
        type='ConvNeXt',
        arch='tiny',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.0,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint, prefix='backbone.'),
    ),
    neck=dict(
        type='ProjectionHead',
        in_channels=[96, 192, 384, 768],
        out_channels=64,
        num_outs=4
    ),
    up_sampler=dict(
        type='MAPUpsampler',
        embed_dim=32,
        num_feat=32,
    ),
    channels=32,
    confidence_num=3,
    num_trans_bins=32,
    align_depths=(1, 1, 1, 1),
    num_kv_frames=[1, 2, 3],
)

HRNet = dict(
    type='HRNet',
    backbone=dict(
        type='ConvNeXt',
        arch='tiny',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.0,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint, prefix='backbone.')
    ),
    neck=dict(
        type='ProjectionHead',
        in_channels=[96, 192, 384, 768],
        out_channels=64,
        num_outs=4
    ),
    up_sampler=dict(
        type='MAPUpsampler',
        embed_dim=32,
        num_feat=32,
    ),
    num_stages=4,
    window_size=32,
)

model = dict(
    type='IMAP',
    generator=dict(
        type='IMAPNet',
        physic_based_dehazer=CMAPNet,
        histogram_refinement=HRNet,
    ),

    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
)

data = dict(
    train_dataloader=dict(samples_per_gpu=2, drop_last=True),
)

# runtime settings
work_dir = f'/root/autodl-tmp/work_dirs/{exp_name}'
