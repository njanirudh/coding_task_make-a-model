#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file is covered by the LICENSE file in the root of this project.

from __future__ import annotations

import typing

import numpy as np
import cv2

from piutils import pi_log

from . import pi_parser
from . import pi_dataset

logger = pi_log.get_logger(__name__)


class PiParserInstanceSegmentation(pi_parser.PiParserBase):
    """A dataset parse for a instance segmentation, object detcetion or
    semantic segmentation network.

    Usage:
        See examples/parser_instance_segmentation.py.
    """

    def make_target(
        self,
        imap: np.ndarray,
        annotations: typing.Dict,
        geometry_transforms: typing.List[pidata.pi_transform.PiRandomTransform],
    ):
        """
        Returns:
            A dictionary with keys as indicated in 'required_targets' in the config
            dictionary passed to the parser:

            'semantics':
                A numpy.ndarray of shape (height, width), dtype numpy.int64
                with semantic label IDs. IDs start from 0 and range to
                <number of semantic labels> as given in the config dictionary
                passed to the parser.

                ID 0 marks Soil/Background pixels.

                The highest ID == <number of semantic classes> marks pixels
                to be ignored (if applicable for the model):

                    0 -> first semantic label in config (should be Soil/Background),
                    1 -> second semantic label in config,
                    2 -> third semantic label in config,

                    ...

                    <number of semantic classes> -> pixels to be ignored

            'boxes':
                A numpy.ndarray of shape (num_instances, 4) and dtype np.float32.
                Bounding box coordinates of each instance in order x0, y0, x1, y1.

            'labels':
                A numpy.ndarray of shape (num_instances,) and dtype np.int64.
                The semantic label ID per instance.
                Also see 'semantics'.

            'area':
                A numpy.ndarray of shape (num_instances,) and dtype np.float32.
                The area of each instance's bounding box.
                Can be used for evaluation.

            'iscrowd':
                A numpy.ndarray of shape (num_instances,) and dtype np.uint8.
                COCO-like. For now, always 0.

            'masks':
                A numpy.ndarray of shape (num_instances, height, width) and dtype np.uint8.
                The binary mask of each instance.

            'keypoints':
                A numpy.ndarray of shape (num_instances, 1, 3) and dtype np.uint8.
                Stem keypoint positions in x, y, visibility.
                If visibility is 0, x and y are also 0.
        """

        imap_ids = np.unique(imap)

        if self._config["required_targets"]["semantics"]:
            semantics = np.zeros(
                (
                    self._output_height // self._output_stride_y,
                    self._output_width // self._output_stride_x,
                ),
                dtype=np.int64,
            )

        # per-instance annotations

        if self._config["required_targets"]["boxes"]:
            boxes = []

        if self._config["required_targets"]["labels"]:
            labels = []

        if self._config["required_targets"]["area"]:
            area = []

        if self._config["required_targets"]["iscrowd"]:
            iscrowd = []

        if self._config["required_targets"]["masks"]:
            masks = []

        if self._config["required_targets"]["keypoints"]:
            keypoints = []

        max_x = np.floor(self._output_width / self._output_stride_x)
        max_y = np.floor(self._output_width / self._output_stride_y)

        for annotation_object_id, annotation_object in annotations.items():
            if (
                annotation_object["type"] in ["segment", "instance"]
                and "imapIds" in annotation_object
                and any(
                    (
                        annotation_imap_id in imap_ids
                        for annotation_imap_id in annotation_object["imapIds"]
                    )
                )
            ):
                annotation_mask = np.isin(imap, annotation_object["imapIds"])

                if self._config["required_targets"]["boxes"]:
                    semantic_label_name = annotation_object["semanticLabelName"]
                    semantic_label_index = self.semantic_labels[semantic_label_name][
                        "index"
                    ]
                    semantics[annotation_mask] = semantic_label_index

                if annotation_object["type"] in ["instance"]:
                    annotation_mask_uint8 = 255 * annotation_mask.astype(np.uint8)

                    if (
                        self._config["required_targets"]["boxes"]
                        or self._config["required_targets"]["area"]
                    ):
                        box_x, box_y, box_width, box_height = cv2.boundingRect(
                            annotation_mask_uint8.astype(np.uint8)
                        )

                        box_x0 = np.clip(box_x, 0.0, max_x)
                        box_x1 = np.clip(box_x + box_width, 0.0, max_x)
                        box_y0 = np.clip(box_y, 0.0, max_y)
                        box_y1 = np.clip(box_y + box_height, 0.0, max_y)

                        box = np.asarray([box_x0, box_y0, box_x1, box_y1])

                        if self._config["required_targets"]["boxes"]:
                            boxes.append(box)

                        if self._config["required_targets"]["area"]:
                            area.append((box[2] - box[0]) * (box[3] - box[1]))

                    if self._config["required_targets"]["labels"]:
                        labels.append(semantic_label_index)

                    if self._config["required_targets"]["iscrowd"]:
                        iscrowd.append(0)

                    if self._config["required_targets"]["masks"]:
                        masks.append(annotation_mask_uint8)

                    if self._config["required_targets"]["keypoints"]:
                        if (
                            "keypoints" in annotation_object
                            and annotation_object["keypoints"]
                        ):
                            keypoint_position = (
                                np.asarray(
                                    annotation_object["keypoints"][0]["coordinates"],
                                    dtype=np.float32,
                                )
                                - np.asarray(
                                    [self._output_offset_x, self._output_offset_y],
                                    dtype=np.float32,
                                )
                            ) / np.asarray(
                                [self._output_stride_x, self._output_stride_y],
                                dtype=np.float32,
                            )

                            keypoint_is_visible = (
                                keypoint_position[0] >= 0
                                and keypoint_position[0] < self._output_width
                                and keypoint_position[1] >= 0.0
                                and keypoint_position[1] < self._output_height
                            )
                        else:
                            keypoint_is_visible = False

                        if keypoint_is_visible:
                            keypoints.append(
                                np.asarray(
                                    [[keypoint_position[0], keypoint_position[1], 1.0]],
                                    dtype=np.float32,
                                )
                            )
                        else:
                            keypoints.append(
                                np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)
                            )

        semantics[imap == pi_dataset.IMAP_IGNORE] = len(self.semantic_labels)

        boxes = np.asarray(boxes, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)
        area = np.asarray(area, dtype=np.float32)
        iscrowd = np.asarray(iscrowd, dtype=np.uint8)
        masks = np.asarray(masks, dtype=np.uint8)
        keypoints = np.asarray(keypoints, dtype=np.float32)

        return dict(
            semantics=semantics,
            boxes=boxes,
            labels=labels,
            area=area,
            iscrowd=iscrowd,
            masks=masks,
            keypoints=keypoints,
        )
