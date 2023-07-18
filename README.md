# Pair then Relation: Pair-Net for Panoptic Scene Graph Generation

<p align="center">
    <br>
    <a href="https://king159.github.io/" target='_blank'> Jinghao Wang*</a>,&nbsp;
    Zhengyu Wen*,&nbsp;
    <a href="https://lxtgh.github.io/" target='_blank'> Xiangtai Li</a>,&nbsp;
      <a href="http://jingkang50.github.io/" target='_blank'>Jingkang Yang</a>,&nbsp;
      <a href="https://gseancdat.github.io/" target='_blank'>Zujing Guo</a>,&nbsp;
      <a href="https://liuziwei7.github.io/" target='_blank'>Ziwei Liu</a>
    <br>
  S-Lab, Nanyang Technological University
  </p>
</p>

<img scr="./doc/architecture.png" align="center" width="60%">

## Abstract

Panoptic Scene Graph (PSG) is a challenging task in Scene Graph Generation (SGG) that aims to create a more comprehensive scene graph representation using panoptic segmentation instead of boxes. However, current PSG methods have limited performance, which can hinder downstream task development. To improve PSG methods, we conducted an in-depth analysis to identify the bottleneck of the current PSG models, finding that inter-object pair-wise recall is a crucial factor which was ignored by previous PSG methods. Based on this, we present a novel framework: **Pair then Relation (Pair-Net)**, which uses a Pair Proposal Network (PPN) to learn and filter sparse pair-wise relationships between subjects and objects. We also observed the sparse nature of object pairs and used this insight to design a lightweight Matrix Learner within the PPN. Through extensive ablation and analysis, our approach significantly improves upon leveraging the strong segmenter baseline. Notably, our approach achieves new state-of-the-art results on the PSG benchmark, with over 10% absolute gains compared to PSGFormer.

## Codebase Structure

``` bash
├── configs
├── data
│   ├── coco
│   │   ├── annotations
|   │   │   ├── panoptic_train2017
|   │   │   ├── panoptic_val2017
│   │   ├── train2017
│   │   └── val2017
│   └── psg
│       ├── psg.json
├── pretrain
├── pairnet
├── scripts
├── tools
│   ├── train.py
│   ├── test.py
├── work_dirs
├── ...
```

## Environment

``` bash
conda install pytorch==1.13.1 torchvision pytorch-cuda=11.7 -c pytorch -c nvidia -y
yes | pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html
conda install scipy -c conda-forge -y
yes | pip install mmcls==0.23.2
yes | pip install mmdet==2.25.1
yes | pip install git+https://github.com/facebookresearch/detectron2.git
yes | pip install git+https://github.com/cocodataset/panopticapi.git
yes | pip install wandb
```

## Training

```bash
#single GPU
PYTHONPATH='.':$PYTHONPATH python configs/mask2former/pairnet.py

#multi GPU
PYTHONPATH='.':$PYTHONPATH bash tools/dist_train.sh configs/mask2former/pairnet.py 4
```

## Testing

```bash
PYTHONPATH='.':$PYTHONPATH \
python tools/test.py \
    configs/deformable_detr/od_r101_vg.py \
    pretrain/deformable_detr_r101_vg.pth \
    --eval bbox

PYTHONPATH='.':$PYTHONPATH \
python tools/test.py \
    configs/mask2former/pairnet.py \
    work_dirs/3090_simpleconv/latest.pth \
    --eval sgdet

```

## Acknowledgements

Pair-Net is developed based on [MMDetection](https://github.com/open-mmlab/mmdetection) and [OpenPSG](https://github.com/Jingkang50/OpenPSG). We sincerely appreciate the efforts of the developers from the previous codebases.
