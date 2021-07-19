#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file is covered by the LICENSE file in the root of this project.

import pathlib

import torch
import numpy as np
import cv2

from piutils import pi_log
from piutils import pi_drawing
from pidata import pi_parser_instance_segmentation

logger = pi_log.get_logger(__name__)

if __name__ == "__main__":

    batch_size = 1  # TODO adjust as required
    num_steps_per_epoch = 100  # TODO adjust as required
    num_epochs = 10  # TODO adjust as required
    num_val_batches = 10  # TODO adjust as required
    num_test_batches = 10  # TODO adjust as required

    num_train_samples = batch_size * num_steps_per_epoch * num_epochs
    num_val_samples = batch_size * num_val_batches
    num_test_samples = batch_size * num_test_batches

    dataset_path_train = pathlib.Path("TODO/TODO")
    dataset_path_val = pathlib.Path("TODO/TODO")
    dataset_path_test = pathlib.Path("TODO/TODO")

    visual = True  # show some training samples, see below

    parser_config = {
        "size": {
            "train": num_train_samples,
            "val": num_val_samples,
            "test": num_test_samples,
        },
        "model_input": {
            "height": 224,  # TODO adjust as required
            "width": 224,
        },
        "input_layers": [
            {
                "channels": 3,
                "mean": [
                    0.0,
                    0.0,
                    0.0,
                ],  # normalization of input, TODO adjust as required
                "name": "rgb",
                "std": [1.0, 1.0, 1.0],  # TODO adjust as required
            }
        ],
        "model_output": {
            "height": 224,  # TODO adjust as required
            "width": 224,  # TODO adjust as required
            "offset_x": 0,
            "offset_y": 0,
            "stride_x": 1,
            "stride_y": 1,
        },
        "required_targets": {
            "semantics": True,  # targets to train the network, TODO adjust as required
            "boxes": True,
            "labels": True,
            "area": True,
            "iscrowd": True,
            "masks": True,
            "keypoints": True,
        },
        "datasets": {
            "train": [
                {"path": dataset_path_train, "sampling_weight": 1.0}
            ],  # TODO add more datasets to list if required
            "val": [{"path": dataset_path_val, "sampling_weight": 1.0}],
            "test": [{"path": dataset_path_train, "sampling_weight": 1.0}],
        },
        "samplers": {
            "train": [
                {
                    "uniform": {
                        "offset_from_boundary_x": 0.0,
                        "offset_from_boundary_y": 0.0,
                        "weight": 0.5,
                    }  # samples a patch from anywhere within the image
                },
                {
                    "instances": {"weight": 0.5}
                },  # samples a patch from image regions with instances, the sampling probability ('sampling_weight') of each class is defined in the 'semantic_labels' section
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
        "seed": {"train": 0, "val": 1, "test": 2},
        "transforms": {  # random transforms per split for data augmentation
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
                        "translation_x_max": 112.0,
                        "translation_x_min": -112.0,
                        "translation_y_max": 112.0,
                        "translation_y_min": -112.0,
                    }
                },
                {
                    "hsv": {
                        "channels": [0, 1, 2],
                        "hue_max": 0.05,
                        "hue_min": -0.05,
                        "saturation_max": 0.05,
                        "saturation_min": -0.05,
                        "value_max": 0.1,
                        "value_min": -0.1,
                        "probability": 1.0,
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
                        "probability": 1.0,
                    }
                },
            ],
            "val": {},  # no random transorms for valdation and test case
            "test": {},
        },
        "semantic_labels": [
            {
                "name": "plant.Soil",
                "has_instances": False,
                "sampling_weight": 1.0,
                "join_with": [],
                "color": [0, 0, 0],
            },
            {
                "name": "plant.Sugarbeet",
                "has_instances": True,
                "sampling_weight": 1.0,  # used for 'instances' sampler (see above)
                "join_with": [],
                "color": [0, 255, 0],
            },
            {
                "name": "plant.Weed",
                "has_instances": True,
                "sampling_weight": 1.0,
                "join_with": [],
                "color": [255, 0, 0],
            },
        ],
    }

    train_data_parser = pi_parser_instance_segmentation.PiParserInstanceSegmentation(
        config=parser_config, split_name="train"
    )

    drawing_kwargs = dict(
        mean=train_data_parser.mean,
        std=train_data_parser.std,
        semantic_labels=train_data_parser.semantic_labels,
        output_width=train_data_parser.output_width,
        output_height=train_data_parser.output_height,
        output_offset_x=train_data_parser.output_offset_x,
        output_offset_y=train_data_parser.output_offset_y,
        output_stride_x=train_data_parser.output_stride_x,
        output_stride_y=train_data_parser.output_stride_y,
        scale_factor=3.0,
    )

    for index, (input_tensor, target_dict) in enumerate(train_data_parser):

        logger.info(f"Sample {index+1}/{len(train_data_parser)}.")
        logger.info(f"    Input shape: {input_tensor.shape}")
        logger.info(f"    Sampled image IDs: {target_dict['image_id']}")
        logger.info(f"    Targets:")

        for target_name, target in target_dict.items():
            logger.info(f"     * '{target_name}':")
            logger.info(f"           Shape: {target.shape}")

        if visual:
            drawing_input = pi_drawing.draw_input(
                input_tensor=input_tensor,
                **drawing_kwargs,
            )
            cv2.imshow(
                "input",
                drawing_input[..., ::-1],  # RGB to BGR
            )

            if "semantics" in target_dict:
                drawing_semantic_labels = pi_drawing.color_semantic_labels(
                    semantics_tensor=target_dict["semantics"],
                    **drawing_kwargs,
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
                drawing_instances = pi_drawing.draw_instances(
                    input_tensor=input_tensor,
                    boxes=target_dict["boxes"] if "boxes" in target_dict else None,
                    keypoints=target_dict["keypoints"]
                    if "keypoints" in target_dict
                    else None,
                    masks=target_dict["masks"] if "masks" in target_dict else None,
                    labels=target_dict["labels"],
                    **drawing_kwargs,
                )
                cv2.imshow(
                    "instances",
                    drawing_instances[..., ::-1],  # RGB to BGR
                )

            cv2.waitKey(100)
