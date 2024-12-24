_base_ = [
    '../_base_/datasets/revide_local.py',
    '../_base_/default_runtime.py',
    './imapnet_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]

exp_name = 'imapnet_revide_40k'

checkpoint = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-tiny_3rdparty_32xb128-noema_in1k_20220301-795e9634.pth'  # noqa

# model settings
MAPNet = dict(
    type='MAPNet',
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
    upsampler=dict(
        type='MAPUpsampler',
        embed_dim=32,
        num_feat=32,
    ),
    channels=32,
    num_trans_bins=32,
    align_depths=(1, 1, 1, 1),
    num_kv_frames=[1, 2, 3],
)

FFA = dict(
    type='FFA',
    gps = 3,
    blocks = 3,#19
)

model = dict(
    type='IMAP',
    generator=dict(
        type='IMAPNet',
        physic_based_dehazer=MAPNet,
        histogram_refinement=FFA,
    ),

    pixel_loss=dict(type='MSE_LOSS', loss_weight=1.0, reduction='mean'),
)

data = dict(
    train_dataloader=dict(samples_per_gpu=1, drop_last=True),
)

# runtime settings
# work_dir = f'/root/autodl-tmp/work_dirs/{exp_name}'
work_dir = f'./work_dirs/{exp_name}'
