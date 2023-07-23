import math

import numpy as np
import torch


class BalancedRelationDataset:
    def __init__(self, dataset, oversample_thr, dict_path, filter_empty_gt=True):
        self.dataset = dataset
        self.oversample_thr = oversample_thr
        self.rel_cls_freq = torch.load(dict_path)
        self.filter_empty_gt = filter_empty_gt
        self.CLASSES = dataset.CLASSES
        self.PALETTE = getattr(dataset, "PALETTE", None)

        repeat_factors = self._get_repeat_factors(dataset, oversample_thr)
        repeat_indices = []
        for dataset_idx, repeat_factor in enumerate(repeat_factors):
            repeat_indices.extend([dataset_idx] * math.ceil(repeat_factor))
        self.repeat_indices = repeat_indices

        flags = []
        if hasattr(self.dataset, "flag"):
            for flag, repeat_factor in zip(self.dataset.flag, repeat_factors):
                flags.extend([flag] * int(math.ceil(repeat_factor)))
            assert len(flags) == len(repeat_indices)
        self.flag = np.asarray(flags, dtype=np.uint8)

    def _get_repeat_factors(self, dataset, repeat_thr):
        """Get repeat factor for each images in the dataset.

        Args:
            dataset (:obj:`CustomDataset`): The dataset
            repeat_thr (float): The threshold of frequency. If an image
                contains the categories whose frequency below the threshold,
                it would be repeated.

        Returns:
            list[float]: The repeat factors for each images in the dataset.
        """

        # 1. For each category c, compute the fraction # of images
        #   that contain it: f(c)

        num_images = len(dataset)

        # rel_cls_freq = torch.load('work_dirs/checkpoints/psg_distribution.pt')
        rel_cls_freq = self.rel_cls_freq
        rel_cls_sum = 0
        for _, freq in rel_cls_freq.items():
            rel_cls_sum += freq

        # 2. For each category c, compute the category-level repeat factor:
        #    r(c) = max(1, sqrt(t/f(c)))
        rel_cls_repeat = {
            cls_id: max(1.0, math.sqrt(repeat_thr / (cls_freq / rel_cls_sum)))
            for cls_id, cls_freq in rel_cls_freq.items()
        }

        # 3. For each image I, compute the image-level repeat factor:
        #    r(I) = max_{c in I} r(c)
        repeat_factors = []
        for idx in range(num_images):
            # rel_cls_ids = set(self.dataset.get_cat_ids(idx))
            rels = self.dataset.get_ann_info(idx)["rels"]

            if len(rels) == 0 and not self.filter_empty_gt:
                rels = set([0, 0, 0])

            repeat_factor = 1
            if len(rels) > 0:
                repeat_factor = max({rel_cls_repeat[rel[2] - 1] for rel in rels})
            repeat_factors.append(repeat_factor)

        return repeat_factors

    def __getitem__(self, idx):
        ori_index = self.repeat_indices[idx]
        return self.dataset[ori_index]

    def get_ann_info(self, idx):
        """Get annotation of dataset by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        ori_index = self.repeat_indices[idx]
        return self.dataset.get_ann_info(ori_index)

    def __len__(self):
        """Length after repetition."""
        return len(self.repeat_indices)
