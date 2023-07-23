import os.path as osp
from collections import defaultdict

import mmcv
import numpy as np
import torch
from detectron2.data.detection_utils import read_image
from mmdet.datasets import DATASETS, CocoDataset
from mmdet.datasets.pipelines import Compose
from pairnet.evaluation import sgg_evaluation
from pairnet.models.relation_heads.approaches import Result
from panopticapi.utils import rgb2id


@DATASETS.register_module()
class OIV6Dataset(CocoDataset):
    def __init__(
        self,
        ann_file,
        pipeline,
        classes=None,
        data_root=None,
        img_prefix="",
        seg_prefix=None,
        proposal_file=None,
        test_mode=False,
        filter_empty_gt=True,
        file_client_args=dict(backend="disk"),
        # New args
        split: str = "train",  # {"train", "val", "test"}
        all_bboxes: bool = False,  # load all bboxes (thing, stuff) for SG
    ):
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.file_client = mmcv.FileClient(**file_client_args)

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
            if not (self.proposal_file is None or osp.isabs(self.proposal_file)):
                self.proposal_file = osp.join(self.data_root, self.proposal_file)

        self.proposal_file = None
        self.proposals = None

        self.all_bboxes = all_bboxes
        self.split = split

        # Load dataset
        dataset = mmcv.load(ann_file)

        for d in dataset["data"]:
            # NOTE: 0-index for object class labels

            # NOTE: 1-index for predicate class labels
            for r in d["relations"]:
                r[2] += 1

        # NOTE: Filter out images with zero relations
        dataset["data"] = [d for d in dataset["data"] if len(d["relations"]) != 0]

        # Get split
        if split == "train":
            self.data = [
                d
                for d in dataset["data"]
                if d["image_id"]
                not in set(dataset["val_image_ids"] + dataset["test_image_ids"])
            ]
        elif split == "val":
            self.data = [
                d for d in dataset["data"] if d["image_id"] in dataset["val_image_ids"]
            ]
        elif split == "test":
            self.data = [
                d for d in dataset["data"] if d["image_id"] in dataset["test_image_ids"]
            ]
        # import pdb; pdb.set_trace();
        # self.data = self.data[:2]
        # Init image infos
        self.data_infos = []
        for d in self.data:
            self.data_infos.append(
                {
                    "filename": d["file_name"],
                    "height": d["height"],
                    "width": d["width"],
                    "id": d["image_id"],
                }
            )
        self.img_ids = [d["id"] for d in self.data_infos]

        # Define classes, 0-index
        # NOTE: Class ids should range from 0 to (num_classes - 1)
        self.THING_CLASSES = dataset["object_classes"]
        self.CLASSES = self.THING_CLASSES
        self.PREDICATES = dataset["relation_classes"]

        # processing pipeline
        self.pipeline = Compose(pipeline)

        if not self.test_mode:
            self._set_group_flag()

    def get_ann_info(self, idx):
        d = self.data[idx]

        # Process bbox annotations
        gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        if self.all_bboxes:
            # NOTE: Get all the bbox annotations (thing + stuff)
            gt_bboxes = np.array(
                [a["bbox"] for a in d["annotations"]], dtype=np.float32
            )
            gt_labels = np.array(
                [a["category_id"] for a in d["annotations"]], dtype=np.int64
            )

        else:
            gt_bboxes = []
            gt_labels = []

            # FIXME: Do we have to filter out `is_crowd`?
            # Do not train on `is_crowd`,
            # i.e just follow the mmdet dataset classes
            # Or treat them as stuff classes?
            # Can try and train on datasets with iscrowd
            # and without and see the difference

            for a in d["annotations"]:
                # NOTE: Only thing bboxes are loaded
                gt_bboxes.append(a["bbox"])
                gt_labels.append(a["category_id"])

            if gt_bboxes:
                gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
                gt_labels = np.array(gt_labels, dtype=np.int64)
            else:
                gt_bboxes = np.zeros((0, 4), dtype=np.float32)
                gt_labels = np.array([], dtype=np.int64)

        # Process relationship annotations
        gt_rels = d["relations"].copy()

        # Filter out dupes!
        if self.split == "train":
            all_rel_sets = defaultdict(list)
            for o0, o1, r in gt_rels:
                all_rel_sets[(o0, o1)].append(r)
            gt_rels = [
                (k[0], k[1], np.random.choice(v)) for k, v in all_rel_sets.items()
            ]
            gt_rels = np.array(gt_rels, dtype=np.int32)
        else:
            # for test or val set, filter the duplicate triplets,
            # but allow multiple labels for each pair
            all_rel_sets = []
            for o0, o1, r in gt_rels:
                if (o0, o1, r) not in all_rel_sets:
                    all_rel_sets.append((o0, o1, r))
            gt_rels = np.array(all_rel_sets, dtype=np.int32)

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            rels=gt_rels,
            rel_maps=None,
            bboxes_ignore=gt_bboxes_ignore,
            masks=None,
        )

        return ann

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        super().pre_pipeline(results)

        results["rel_fields"] = []

    def prepare_test_img(self, idx):
        # For SGG, since the forward process may need gt_bboxes/gt_labels,
        # we should also load annotation as if in the training mode.
        return self.prepare_train_img(idx)

    def evaluate(
        self,
        results,
        metric="predcls",
        logger=None,
        jsonfile_prefix=None,
        classwise=True,
        multiple_preds=False,
        iou_thrs=0.5,
        nogc_thres_num=None,
        detection_method="bbox",
        **kwargs,
    ):
        """Overwritten evaluate API:

        For each metric in metrics, it checks whether to invoke ps or sg
        evaluation. if the metric is not 'sg', the evaluate method of super
        class is invoked to perform Panoptic Segmentation evaluation. else,
        perform scene graph evaluation.
        """
        assert detection_method == "bbox", "Open Images V6 has no segmentation now."
        assert metric not in ["PQ"], "Open Images V6 has no segmentation now."
        metrics = metric if isinstance(metric, list) else [metric]

        # Available metrics
        allowed_sg_metrics = ["predcls", "sgcls", "sgdet"]
        allowed_od_metrics = ["PQ"]

        sg_metrics, od_metrics = [], []
        for m in metrics:
            if m in allowed_od_metrics:
                od_metrics.append(m)
            elif m in allowed_sg_metrics:
                sg_metrics.append(m)
            else:
                raise ValueError("Unknown metric {}.".format(m))

        if len(od_metrics) > 0:
            # invoke object detection evaluation.
            # Temporarily for bbox
            if not isinstance(results[0], Result):
                # it may be the results from the son classes
                od_results = results
            else:
                od_results = [{"pan_results": r.pan_results} for r in results]
            return super().evaluate(
                od_results,
                metric,
                logger,
                jsonfile_prefix,
                classwise=classwise,
                **kwargs,
            )

        if len(sg_metrics) > 0:
            """Invoke scene graph evaluation.

            prepare the groundtruth and predictions. Transform the predictions
            of key-wise to image-wise. Both the value in gt_results and
            det_results are numpy array.
            """
            if not hasattr(self, "test_gt_results"):
                print("\nLoading testing groundtruth...\n")
                prog_bar = mmcv.ProgressBar(len(self))
                gt_results = []
                for i in range(len(self)):
                    ann = self.get_ann_info(i)

                    # NOTE: Change to object class labels 1-index here
                    ann["labels"] += 1
                    gt_masks = None
                    # load gt pan seg masks done
                    # labels_coco.sort()
                    # print('\ngt_labels: ', labels_coco)

                    gt_results.append(
                        Result(
                            bboxes=ann["bboxes"],
                            labels=ann["labels"],
                            rels=ann["rels"],
                            relmaps=ann["rel_maps"],
                            rel_pair_idxes=ann["rels"][:, :2],
                            rel_labels=ann["rels"][:, -1],
                            masks=gt_masks,
                        )
                    )
                    prog_bar.update()

                print("\n")
                self.test_gt_results = gt_results

            return sgg_evaluation(
                sg_metrics,
                groundtruths=self.test_gt_results,
                predictions=results,
                iou_thrs=iou_thrs,
                logger=logger,
                ind_to_predicates=["__background__"] + self.PREDICATES,
                multiple_preds=multiple_preds,
                # predicate_freq=self.predicate_freq,
                nogc_thres_num=nogc_thres_num,
                detection_method=detection_method,
            )

    def get_statistics(self):
        freq_matrix = self.get_freq_matrix()
        eps = 1e-3
        freq_matrix += eps
        pred_dist = np.log(freq_matrix / freq_matrix.sum(2)[:, :, None] + eps)

        result = {
            "freq_matrix": torch.from_numpy(freq_matrix),
            "pred_dist": torch.from_numpy(pred_dist).float(),
        }
        if result["pred_dist"].isnan().any():
            print("check pred_dist: nan")
        return result

    def get_freq_matrix(self):
        num_obj_classes = len(self.CLASSES)
        num_rel_classes = len(self.PREDICATES)

        freq_matrix = np.zeros(
            (num_obj_classes, num_obj_classes, num_rel_classes + 1), dtype=np.float
        )
        progbar = mmcv.ProgressBar(len(self.data))

        for d in self.data:
            segments = d["segments_info"]
            relations = d["relations"]

            for rel in relations:
                object_index = segments[rel[0]]["category_id"]
                subject_index = segments[rel[1]]["category_id"]
                relation_index = rel[2]

                freq_matrix[object_index, subject_index, relation_index] += 1

            progbar.update()

        return freq_matrix
