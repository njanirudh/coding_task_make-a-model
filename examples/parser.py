#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file is covered by the LICENSE file in the root of this project.

from __future__ import annotations

import cv2

from src.pidata.pidata import pi_parser
from src.piutils.piutils import pi_drawing
from src.piutils.piutils import pi_log

logger = pi_log.get_logger(__name__)

if __name__ == "__main__":

    parser_config = {
        "datasets": {
            "train": [
                {
                    "path": "/home/anirudh/NJ/Interview/Pheno-Inspect/"
                            "coding_task_make-a-model/dataset/sugarbeet_weed_dataset",  # TODO adjust as required
                    "sampling_weight": 1.0,
                }
            ],
            "val": [{"path": "TODO/sugarbeet_weed_dataset", "sampling_weight": 1.0}],
            "test": [{"path": "TODO/sugarbeet_weed_dataset", "sampling_weight": 1.0}],
        },
        "input_layers": [
            {
                "name": "rgb",
                "channels": 3,
                "mean": [0.319, 0.3992, 0.4686],  # input data normalization
                "std": [0.1768, 0.2009, 0.2138],
            }
        ],
        "instance_filter": {"min_box_area": 100, "min_mask_area": 100},
        "model_input": {"height": 448, "width": 448},  # TODO adjust as required
        "model_output": {
            "height": 448,  # TODO adjust as required
            "offset_x": 0,
            "offset_y": 0,
            "stride_x": 1,
            "stride_y": 1,
            "width": 448,
        },
        "required_targets": {
            "area": True,
            "boxes": True,
            "iscrowd": True,
            "keypoints": True,
            "labels": True,
            "masks": True,
            "semantics": True,
        },
        "samplers": {
            "train": [
                {
                    "uniform": {
                        "offset_from_boundary_x": 0.0,
                        "offset_from_boundary_y": 0.0,
                        "weight": 0.5,
                    }
                },  # samples a patch from anywhere within the image
                {"instances": {"weight": 0.5}},
                # samples a patch from image regions with plant instances
                # the frequence of each class is determined by the 'sampling_weight',
                # see semantic labels
            ],
            "val": [
                {
                    "uniform": {
                        "offset_from_boundary_x": 0.0,
                        "offset_from_boundary_y": 0.0,
                        "weight": 0.5,
                    }
                },
                {"instances": {"weight": 0.5}},
            ],
            "test": [
                {
                    "uniform": {
                        "offset_from_boundary_x": 0.0,
                        "offset_from_boundary_y": 0.0,
                        "weight": 0.5,
                    }
                },
                {"instances": {"weight": 0.5}},
            ],
        },
        "seed": {"test": 2, "train": 0, "val": 1},
        "semantic_labels": [
            {
                "color": [0, 0, 0],  # color for visualization
                "has_instances": False,
                "join_with": [
                    "plant.Unlabeled",
                    "plant.Vegetation",
                ],  # unrecognized vegetation is mapped to Soil/Background
                "name": "plant.Soil",
                "sampling_weight": 1.0,
            },
            {
                "color": [0, 255, 0],
                "has_instances": True,
                "join_with": [],
                "name": "plant.Sugarbeet",
                "sampling_weight": 1.0,
            },
            {
                "color": [255, 0, 0],
                "has_instances": True,
                "join_with": [],
                "name": "plant.Weed",
                "sampling_weight": 1.0,
            },
        ],
        "transforms": {
            "train": [
                {
                    "affine": {
                        "flip_x_probability": 0.5,
                        "flip_y_probability": 0.5,
                        "probability": 1.0,
                        "rotation_max": 3.141592653589793,
                        "rotation_min": -3.141592653589793,
                        "scaling_x_max": 1.25,
                        "scaling_x_min": 0.8,
                        "scaling_y_max": 1.25,
                        "scaling_y_min": 0.8,
                        "shearing_x_max": 0.2,
                        "shearing_x_min": -0.2,
                        "shearing_y_max": 0.2,
                        "shearing_y_min": -0.2,
                        "translation_x_max": 224.0,
                        "translation_x_min": -224.0,
                        "translation_y_max": 224.0,
                        "translation_y_min": -224.0,
                    }
                },
                {
                    "hsv": {
                        "channels": [0, 1, 2],
                        "hue_max": 0.05,
                        "hue_min": -0.05,
                        "probability": 1.0,
                        "saturation_max": 0.05,
                        "saturation_min": -0.05,
                        "value_max": 0.1,
                        "value_min": -0.1,
                    }
                },
                {
                    "contrast": {
                        "channels": [0, 1, 2],
                        "contrast_max": 0.1,
                        "contrast_min": -0.1,
                        "probability": 1.0,
                    }
                },
                {
                    "blur": {
                        "blur_max": 2.0,
                        "blur_min": 0.0,
                        "channels": [0, 1, 2],
                        "probability": 0.2,
                    }
                },
            ],
            "val": {},  # no random transorms for valdation and test case
            "test": {},
        },
    }

    train_data_parser = pi_parser.PiParser(
        config=parser_config,
        split_name="train",
        num_samples=100,  # number of samples to be drawn, set to a multiple of teh batch size
        numpy_to_tensor_func=None,
        # framework-dependent, e.g. torch.from_numpy (PyTorch), if None, the returned type is numpy.ndarray
    )

    drawing_kwargs = dict(
        mean=train_data_parser.mean,  # undo input normalization
        std=train_data_parser.std,
        semantic_labels=train_data_parser.semantic_labels,
        output_width=train_data_parser.output_width,
        # position of network output wrt to input, see pidata.pi_parser.__init__()
        output_height=train_data_parser.output_height,
        output_offset_x=train_data_parser.output_offset_x,
        output_offset_y=train_data_parser.output_offset_y,
        output_stride_x=train_data_parser.output_stride_x,
        output_stride_y=train_data_parser.output_stride_y,
        scale_factor=1.0,
    )

    for sample_index, (input_tensor, target_dict) in enumerate(train_data_parser):
        if sample_index != 15:
            continue
        logger.info(f"Sample {sample_index + 1}/{len(train_data_parser)}.")
        logger.info(f"    Input shape: {input_tensor.shape}")
        logger.info(f"    Sampled image IDs: {target_dict['image_id']}")

        logger.info(f"    Targets:")
        for target_name, target in target_dict.items():
            logger.info(f"     * '{target_name}':")
            logger.info(f"       Shape: {target.shape}")

        drawing_input = pi_drawing.draw_input(
            input_tensor=input_tensor,
            **drawing_kwargs,
        )
        cv2.imshow(
            "input",
            drawing_input[..., ::-1],  # RGB to BGR
        )

        if "semantics" in target_dict:
            drawing_semantic_labels = (
                pi_drawing.color_semantic_labels(  # feel free to use in you own code
                    semantics_tensor=target_dict["semantics"],
                    **drawing_kwargs,
                )
            )
            cv2.imshow(
                "semantic_labels",
                drawing_semantic_labels[..., ::-1],  # RGB to BGR
            )

        if (
                "boxes" in target_dict
                or "keypoints" in target_dict
                or "masks" in target_dict
        ) and "labels" in target_dict:
            drawing_instances = (
                pi_drawing.draw_instances(  # feel free to use in your own code
                    input_tensor=input_tensor,
                    boxes=target_dict["boxes"] if "boxes" in target_dict else None,
                    keypoints=(
                        target_dict["keypoints"] if "keypoints" in target_dict else None
                    ),
                    masks=target_dict["masks"] if "masks" in target_dict else None,
                    labels=target_dict["labels"],
                    **drawing_kwargs,
                )
            )
            cv2.imshow(
                "instances",
                drawing_instances[..., ::-1],  # RGB to BGR
            )

        cv2.waitKey()
