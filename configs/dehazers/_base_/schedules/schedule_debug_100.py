# Short train for net debug
# Only 100 iterations

total_iters = 100
runner = dict(type='IterBasedRunner', max_iters=100)
checkpoint_config = dict(by_epoch=False, interval=100)
# remove gpu_collect=True in non distributed training