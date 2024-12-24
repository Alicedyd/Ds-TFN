## Installation

This implementation is based on [MMEditing](https://github.com/open-mmlab/mmediting),
which is an open-source image and video editing toolbox.

```
python 3.10.9
pytorch 1.12.1
torchvision 0.13.1
cuda 11.3
```

Below are quick steps for installation.

**Step 1.**
Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/).

**Step 2.**
Install MMCV with [MIM](https://github.com/open-mmlab/mim).

```shell
pip3 install openmim
mim install mmcv-full
```

**Step 3.**
Install MAP-Net from source.

```shell
cd IMAP
pip3 install -e .
pip install yapf==0.40.0
```

Please refer to [MMEditing Installation](https://github.com/open-mmlab/mmediting/blob/master/docs/en/install.md) for more detailed instruction.


## Getting Started

You can train IMAP-Net on REVIDE using the below command with 1 GPUs:

```shell
bash tools/dist_train.sh configs/dehazers/cmapnet/cmapnet_revide_debug.py 1
bash tools/dist_train.sh configs/dehazers/cmapnet/cmapnet_revide.py 1
bash tools/dist_train.sh configs/dehazers/cmapnet/cmapnet_revide_80k.py 1
bash tools/dist_train.sh configs/dehazers/cmapnet_single/cmapnet_single_revide_80k.py 1
bash tools/dist_train.sh configs/dehazers/mapnet/mapnet_revide.py 1
bash tools/dist_train.sh configs/dehazers/mapnet/mapnet_revide_80k.py 1
bash tools/dist_train.sh configs/dehazers/imapnet/imapnet_revide.py 1
bash tools/dist_train.sh configs/dehazers/imapnet/imapnet_revide_80k.py 1
```


## Evaluation

We mainly use [psnr and ssim](./mmedit/core/evaluation/metrics.py) to measure the model performance.

You can use the following command with 1 GPU to test your trained model `xxx.pth`:

```shell
bash tools/dist_test.sh configs/dehazers/cmapnet/cmapnet_single_revide_80k.py /root/autodl-tmp/work_dirs/cmapnet_single_revide_80k/iter_80000.pth 1
bash tools/dist_test.sh configs/dehazers/cmapnet/cmapnet_revide_80k.py /root/autodl-tmp/work_dirs/cmapnet_revide_80k/iter_80000.pth 1
bash tools/dist_test.sh configs/dehazers/cmapnet/cmapnet_revide.py /root/autodl-tmp/work_dirs/cmapnet_revide_40k/iter_40000.pth 1
bash tools/dist_test.sh configs/dehazers/mapnet/mapnet_revide.py /root/autodl-tmp/work_dirs/mapnet_revide_40k/iter_40000.pth 1
bash tools/dist_test.sh configs/dehazers/mapnet/mapnet_revide_80k.py /root/autodl-tmp/work_dirs/mapnet_revide_80k/iter_80000.pth 1
bash tools/dist_test.sh configs/dehazers/imapnet/imapnet_revide.py /root/autodl-tmp/work_dirs/imapnet_revide_40k/iter_40000.pth 1
bash tools/dist_test.sh configs/dehazers/imapnet/imapnet_revide_80k.py /root/autodl-tmp/work_dirs/imapnet_revide_80k/iter_80000.pth 1
```
