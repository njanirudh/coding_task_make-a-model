#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file is covered by the LICENSE file in the root of this project.

from __future__ import annotations

import typing
import pathlib

import numpy as np
import cv2
import torch

from piutils import pi_log

from . import pi_dataset
from . import pi_transform

logger = pi_log.get_logger(__name__)


class PiParserBase(torch.utils.data.Dataset):
    """Base class for our dataset parsers."""

    def __init__(self, config: typing.Dict, split_name: str):
        if split_name not in {"train", "val", "test"}:
            raise ValueError("Invalid split name: '{split_name}'")

        self._config = config
        self._split_name = split_name

        self._input_width = self._config["model_input"]["width"]
        self._input_height = self._config["model_input"]["height"]

        self._output_width = self._config["model_output"]["width"]
        self._output_height = self._config["model_output"]["height"]
        self._output_offset_x = self._config["model_output"]["offset_x"]
        self._output_offset_y = self._config["model_output"]["offset_y"]
        self._output_stride_x = self._config["model_output"]["stride_x"]
        self._output_stride_y = self._config["model_output"]["stride_y"]

        self._size = self._config["size"][self._split_name]
        seed = self._config["seed"][self._split_name]

        random = np.random.RandomState(seed)
        self._seeds = random.choice(
            2 ** 32, size=self._size
        )  # one random seed for each item

        # get datasets for this split from config

        datasets_data = [
            dataset_data for dataset_data in self._config["datasets"][self._split_name]
        ]

        dataset_data_per_dataset = {
            pi_dataset.PiDataset(path=pathlib.Path(dataset_data["path"])): dataset_data
            for dataset_data in datasets_data
        }

        # get semantic labels from config

        self._semantic_labels = {
            semantic_label_data["name"]: {
                "sampling_weight": semantic_label_data["sampling_weight"],
                "index": semantic_label_index,
                "join_with": semantic_label_data["join_with"],
                "has_instances": semantic_label_data["has_instances"],
                "color": semantic_label_data["color"],
            }
            for semantic_label_index, semantic_label_data in enumerate(
                self._config["semantic_labels"]
            )
        }

        logger.info("Semantic labels:")
        for semantic_label_name, semantic_label_data in self._semantic_labels.items():
            logger.info(f" * '{semantic_label_name}':")
            logger.info(f"       Index: {semantic_label_data['index']}")
            logger.info(
                f"       Sampling weight: {semantic_label_data['sampling_weight']}"
            )
            logger.info(
                f"       Has instances: {semantic_label_data['sampling_weight']}"
            )

        # prepare the 'join_with' option
        # remap a group of labels to a single class

        self._semantic_labels_mapping = {
            **{
                semantic_label_name: semantic_label_name
                for semantic_label_name in self._semantic_labels
            },
            **{
                semantic_label_to_join: semantic_label_name
                for semantic_label_name, semantic_label_data in self._semantic_labels.items()
                for semantic_label_to_join in semantic_label_data["join_with"]
            },
        }

        logger.info(f"Remap semantic labels '{self._split_name}':")
        for (
            semantic_label_name,
            mapped_to_name,
        ) in self._semantic_labels_mapping.items():
            logger.info(f" * {semantic_label_name} \u2192 {mapped_to_name}")

        # get samplers for this split from config

        samplers_data = [
            {"type": sampler_type, **sampler_data}
            for sampler_dict in self._config["samplers"][self._split_name]
            for sampler_type, sampler_data in sampler_dict.items()
        ]

        if not samplers_data:
            raise ValueError(f"No samplers defined for split: '{self._split_name}'")

        # get valid instances samplings index (list of instances that are allowed to be sampled)
        # if instances sampler is used (some datasets can contain no instance of any semantic class)

        if any((sampler_data["type"] == "instances" for sampler_data in samplers_data)):
            self._instances_sampling_index_data_per_dataset = {}
            self._instances_sampling_weights_per_dataset = {}

            for dataset in dataset_data_per_dataset.keys():
                sampling_index_data = dataset.sampling_index("instances")

                semantic_labels_to_join = [
                    [semantic_label_data["name"]] + semantic_label_data["join_with"]
                    for semantic_label_data in self._config["semantic_labels"]
                    if semantic_label_data["has_instances"]
                ]

                sampling_index_data = {
                    semantic_label_names[0]: [
                        sampling_item
                        for semantic_label_name in semantic_label_names
                        if semantic_label_name in sampling_index_data
                        for sampling_item in sampling_index_data[semantic_label_name]
                    ]
                    for semantic_label_names in semantic_labels_to_join
                }

                # filter out those labels that do not have any items

                sampling_index_data = {
                    semantic_label_name: sampling_item_data
                    for semantic_label_name, sampling_item_data in sampling_index_data.items()
                    if sampling_item_data
                }

                semantic_label_sampling_weights_norm = sum(
                    self._semantic_labels[semantic_label_name]["sampling_weight"]
                    for semantic_label_name in sampling_index_data
                )

                semantic_label_sampling_weights = [
                    self._semantic_labels[semantic_label_name]["sampling_weight"]
                    / semantic_label_sampling_weights_norm
                    for semantic_label_name in sampling_index_data
                ]

                if sampling_index_data and semantic_label_sampling_weights_norm > 0.0:
                    self._instances_sampling_index_data_per_dataset[
                        dataset
                    ] = sampling_index_data
                    self._instances_sampling_weights_per_dataset[
                        dataset
                    ] = semantic_label_sampling_weights
                else:
                    logger.warning(
                        f"No instances to sample in dataset '{dataset.name}', split '{self._split_name}'."
                    )

        # get valid samplers per dataset

        self._sampler_data_per_dataset = {}
        self._sampler_weight_per_dataset = {}

        for dataset in dataset_data_per_dataset.keys():
            if dataset_data_per_dataset[dataset]["sampling_weight"] <= 0.0:
                logger.warning(
                    f"Dataset '{dataset.name}', split '{self._split_name}' with non-positive sampling weight."
                )
                continue

            samplers_data_for_dataset = []
            samplers_weights_for_dataset = []

            for sampler_data in samplers_data:
                accept = False

                if sampler_data["type"] not in ["uniform", "instances"]:
                    raise NotImplementedError(
                        "Sampler '{sampler_data['type']}' not recognized."
                    )

                if sampler_data["type"] in ["uniform"] and sampler_data["weight"] > 0.0:
                    accept = True

                if (
                    sampler_data["type"] in ["instances"]
                    and sampler_data["weight"] > 0.0
                    and dataset
                    in self._instances_sampling_index_data_per_dataset  # exclude if not instances to sample
                ):
                    accept = True

                if accept:
                    samplers_data_for_dataset.append(sampler_data)
                    samplers_weights_for_dataset.append(sampler_data["weight"])
                else:
                    logger.warning(
                        f"Sampler '{sampler_data['type']}' invalid for dataset '{dataset.name}', "
                        f"split '{self._split_name}'."
                    )

            if samplers_data_for_dataset:
                samplers_weights_norm = sum(samplers_weights_for_dataset)

                if samplers_weights_norm <= 0.0:
                    raise ValueError(
                        "Invalid sampling configuration for dataset '{dataset.name}', "
                        f"split '{self._split_name}'."
                    )

                samplers_weights_for_dataset = [
                    weight / samplers_weights_norm
                    for weight in samplers_weights_for_dataset
                ]

                self._sampler_data_per_dataset[dataset] = samplers_data_for_dataset
                self._sampler_weight_per_dataset[dataset] = samplers_weights_for_dataset
            else:
                logger.warning(
                    f"Ignoring dataset '{dataset.name}', "
                    f"split '{self._split_name}' with no valid samplers."
                )

        if not self._sampler_data_per_dataset:
            raise ValueError(
                f"No valid samplers for split '{self._split_name}'. "
                "If samples 'instances' is used as the only sampler, "
                "make sure one of dataset contains instances with the "
                "semantic labels defined in the config."
            )

        # keep those datasets with valid samplers

        self._datasets = [dataset for dataset in self._sampler_data_per_dataset.keys()]
        datasets_sampling_weights = [
            dataset_data_per_dataset[dataset]["sampling_weight"]
            for datset in self._datasets
        ]
        datasets_sampling_weights_norm = sum(datasets_sampling_weights)
        self._datasets_sampling_weights = [
            weight / datasets_sampling_weights_norm
            for weight in datasets_sampling_weights
        ]

        logger.info(
            f"Using {len(self._datasets)} dataset(s) from split '{self._split_name}':"
        )
        for dataset, sampling_weight in zip(
            self._datasets, self._datasets_sampling_weights
        ):
            logger.info(f" * Name: {dataset.name}")
            logger.info(f"   Sampling weight (normalized): {sampling_weight}")
            logger.info(
                f"   Samplers: "
                + str(
                    [
                        sampler_data["type"]
                        for sampler_data in self._sampler_data_per_dataset[dataset]
                    ]
                )
            )
            logger.info(
                f"   Sampler weights (normalized): {self._sampler_weight_per_dataset[dataset]}"
            )

        # get transforms for this split from config

        self._geometry_transforms_data = [
            {"type": transform_type, **transform_data}
            for transform_dict in self._config["transforms"][self._split_name]
            for transform_type, transform_data in transform_dict.items()
            if transform_type in {"affine"}
        ]

        self._color_transforms_data = [
            {"type": transforms_type, **transforms_data}
            for transforms_dict in self._config["transforms"][self._split_name]
            for transforms_type, transforms_data in transforms_dict.items()
            if transforms_type in {"hsv", "contrast", "blur"}
        ]

        logger.info(
            f"Using {len(self._geometry_transforms_data)} "
            f"+ {len(self._color_transforms_data)} random transforms for data augmentation:"
        )
        for transform_data in (
            self._geometry_transforms_data + self._color_transforms_data
        ):
            logger.info(f"    * Type: {transform_data['type']}")

        sample_size = np.max([self._input_width, self._input_height])

        affine_transform_data = next(
            (
                transform_data
                for transform_data in self._geometry_transforms_data
                if transform_data["type"] == "affine"
            ),
            None,
        )

        if affine_transform_data is not None:
            scaling_max = np.max(
                np.absolute(
                    [
                        affine_transform_data["scaling_x_min"],
                        affine_transform_data["scaling_x_max"],
                        affine_transform_data["scaling_y_min"],
                        affine_transform_data["scaling_y_max"],
                    ]
                )
            )

            sample_size *= scaling_max

            shearing_max = np.max(
                np.absolute(
                    [
                        affine_transform_data["shearing_x_min"],
                        affine_transform_data["shearing_x_max"],
                        affine_transform_data["shearing_y_min"],
                        affine_transform_data["shearing_y_max"],
                    ]
                )
            )

            if shearing_max != 0.0:
                sample_size += np.absolute(np.sin(shearing_max) * sample_size)

            translation_max = np.max(
                np.absolute(
                    [
                        affine_transform_data["translation_x_min"],
                        affine_transform_data["translation_x_max"],
                        affine_transform_data["translation_y_min"],
                        affine_transform_data["translation_y_max"],
                    ]
                )
            )

            sample_size += translation_max

            rotation_max = np.max(
                np.absolute(
                    [
                        affine_transform_data["rotation_min"],
                        affine_transform_data["rotation_max"],
                    ]
                )
            )

            if rotation_max != 0.0:
                sample_size *= np.sqrt(2.0)

        sample_size = np.ceil(sample_size).astype(np.int).item()

        self._sample_size = sample_size

        logger.info(f"Drawing samples of size: {self._sample_size}")

        self._geometry_transforms = [
            self._geometry_transform_from_data(transform_data)
            for transform_data in self._geometry_transforms_data
        ]

        self._color_transforms = [
            self._color_transform_from_data(transform_data)
            for transform_data in self._color_transforms_data
        ]

        # get input layers and normalization from config

        mean = self._input_mean_from_config()
        std = self._input_std_from_config()

        input_layer_names = []
        input_channels = 0

        for input_layer_data in self._config["input_layers"]:
            input_layer_names += [input_layer_data["name"]]
            input_channels += input_layer_data["channels"]

        self._input_layer_names = input_layer_names
        self._input_channels = input_channels
        self._mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, -1)
        self._std = np.asarray(std, dtype=np.float32).reshape(1, 1, -1)

        logger.info("Input layers:")
        for input_layer_data in self._config["input_layers"]:
            logger.info(f"    * Name: {input_layer_data['name']}")
            logger.info(f"    * Channels: {input_layer_data['channels']}")

        logger.info(f"Parser initializaton done (split '{self._split_name}').")

    @property
    def config(self) -> typing.Dict:
        return self._config

    @property
    def semantic_labels(self) -> typing.Dict:
        return self._semantic_labels

    @property
    def semantic_labels_mapping(self) -> typing.Dict[str, str]:
        return self._semantic_labels_mapping

    @property
    def mean(self) -> np.ndarray:
        return self._mean

    @property
    def std(self) -> np.ndarray:
        return self._std

    @property
    def output_width(self) -> int:
        return self._output_width

    @property
    def output_height(self) -> int:
        return self._output_height

    @property
    def output_offset_x(self) -> int:
        return self._output_offset_x

    @property
    def output_offset_y(self) -> int:
        return self._output_offset_y

    @property
    def output_stride_x(self) -> int:
        return self._output_stride_x

    @property
    def output_stride_y(self) -> int:
        return self._output_stride_y

    def sample(self, random: np.random.RandomState) -> typing.Dict[str, torch.Tensor]:
        dataset = random.choice(
            self._datasets, p=self._datasets_sampling_weights, replace=False
        )

        if __debug__:
            logger.debug(
                f"Sample from dataset '{dataset.name}' (split '{self._split_name}')."
            )

        sampler_data = random.choice(
            self._sampler_data_per_dataset[dataset],
            p=self._sampler_weight_per_dataset[dataset],
            replace=False,
        )

        if __debug__:
            logger.debug(f"Using sampler '{sampler_data['type']}'.")

        if sampler_data["type"] == "uniform":
            dataset_item_names = list(dataset.items.keys())

            dataset_item_name = random.choice(dataset_item_names, replace=False)

            if __debug__:
                logger.debug(f"Sampling from dataset item '{dataset_item_name}'.")

            dataset_item = dataset.items[dataset_item_name]

            offset_x = sampler_data["offset_from_boundary_x"]
            offset_y = sampler_data["offset_from_boundary_y"]

            if offset_x > dataset_item.width - offset_x:
                raise RuntimeError(
                    "Offset from boundary ({offset_x}) larger than "
                    f"half raster width ({dataset_item.width}): {dataset_item_name}"
                )

            if offset_y > dataset_item.height - offset_y:
                raise RuntimeError(
                    "Offset from boundary ({offset_y}) larger than "
                    f"half raster height ({dataset_item.height}): {dataset_item_name}"
                )

            seed_x = random.uniform(offset_x, dataset_item.width - offset_x)
            seed_y = random.uniform(offset_y, dataset_item.height - offset_y)

        elif sampler_data["type"] == "instances":
            sampling_index_data = self._instances_sampling_index_data_per_dataset[
                dataset
            ]
            semantic_label_sampling_weights = (
                self._instances_sampling_weights_per_dataset[dataset]
            )

            if not sampling_index_data:
                raise ValueError(
                    f"No instances in sampling index of dataset: '{dataset.path}'"
                )

            semantic_label_name = random.choice(
                list(sampling_index_data.keys()),
                p=semantic_label_sampling_weights,
                replace=False,
            )

            sampling_seed_data = random.choice(
                sampling_index_data[semantic_label_name], replace=False
            )

            dataset_item_name = sampling_seed_data["itemName"]

            if __debug__:
                logger.debug(f"Sampling from dataset item '{dataset_item_name}'.")

            dataset_item = dataset.items[dataset_item_name]

            seed_x = sampling_seed_data["coordinates"][0]
            seed_y = sampling_seed_data["coordinates"][1]
        else:
            raise NotImplementedError(
                "Sampler not implemented: '{sampler_data['type']}'"
            )

        seed_x = np.round(seed_x).astype(np.int).item()
        seed_y = np.round(seed_y).astype(np.int).item()

        x = seed_x - self._sample_size // 2
        y = seed_y - self._sample_size // 2
        width = self._sample_size
        height = self._sample_size

        input_raster, imap, annotations = self._query_region_from_map(
            dataset_item=dataset_item, x=x, y=y, width=width, height=height
        )

        # apply transforms

        for geometry_transform in self._geometry_transforms:
            geometry_transform.resample(random=random)

            input_raster = geometry_transform.transform_raster(
                input_raster, fill_value=0.0, interpolation="linear"
            )

            imap = geometry_transform.transform_raster(
                imap, fill_value=pi_dataset.IMAP_IGNORE, interpolation="nearest"
            )

            for annotation_object_id, annotation_object_data in annotations.items():
                if "keypoints" in annotation_object_data:
                    keypoint_positions = np.asarray(
                        [
                            keypoint_data["coordinates"]
                            for keypoint_data in annotation_object_data["keypoints"]
                        ]
                    )

                    if keypoint_positions.size:
                        keypoint_positions = geometry_transform.transform_points(
                            keypoint_positions
                        )
                        annotation_object_data["keypoints"] = [
                            {"coordinates": keypoint_positions[keypoint_index].tolist()}
                            for keypoint_index in range(keypoint_positions.shape[0])
                        ]

        for color_transform in self._color_transforms:
            color_transform.resample(random=random)
            input_raster = color_transform.transform_raster(
                input_raster, fill_value=0.0, interpolation="linear"
            )

        # normalize input

        input_raster = (input_raster.astype(np.float32) - self._mean) / self._std
        input_raster = input_raster.transpose((2, 0, 1))  # channels first

        if __debug__:
            logger.debug(f"Input raster shape: {input_raster.shape}")
            logger.debug("Input raster range (after normalization):")
            for channel_index in range(input_raster.shape[0]):
                input_band = input_raster[channel_index]
                logger.debug(f" * Band {channel_index}:")
                logger.debug(f"       Min: {input_band.min()}")
                logger.debug(f"       Max: {input_band.max()}")
                logger.debug(f"       Mean: {input_band.mean()}")
                logger.debug(f"       Dtype: {input_band.dtype}")

                # debug output
                # cv2.imshow(f"input_band_{channel_index}", input_band)
                # cv2.waitKey()

        # imap to output size of model

        imap = imap[
            self._output_offset_y : (
                self._output_offset_y + self._output_height
            ) : self._output_stride_y,
            self._output_offset_x : (
                self._output_offset_x + self._output_width
            ) : self._output_stride_x,
        ]

        target = self.make_target(
            imap=imap,
            annotations=annotations,
            geometry_transforms=self._geometry_transforms,
        )

        return input_raster, target

    def make_target(
        self,
        imap: np.ndarray,
        annotations: typing.Dict,
        geometry_transforms: typing.List[pidata.pi_transform.PiRandomTransform],
    ) -> typing.Dict[str, np.ndarray]:
        """Parses annotations from dataset as required by task or model.

        To be implemented by derived classes.
        """

    def __len__(self) -> int:
        return self._size

    def __getitem__(
        self, index: int
    ) -> typing.Tuple[np.ndarray, typing.Dict[str, np.ndarray]]:
        """
        Returns:
            A tuple of
                input raster,
                    a numpy.ndarray of dtype numpy.float32, and
                target,
                    a dictionary of numpy.ndarray depending on the
                    implementation of make_target().
        """
        input_raster, target = self.sample(
            random=np.random.RandomState(self._seeds[index])
        )
        target["image_id"] = np.asarray(index, dtype=np.int64)
        return input_raster, target

    def _query_region_from_map(
        self, dataset_item: CiDatasetItemType, x: int, y: int, width: int, height: int
    ) -> typing.Tuple[np.ndarray, np.ndarray, typing.Dict]:
        """
        Returns:
            Tuple of input raster, imap and annotations (not transformed!).
        """
        input_rasters = []

        for input_layer_name in self._input_layer_names:
            raster = dataset_item.map.provide_raster(
                raster_layer_name=input_layer_name, x=x, y=y, width=width, height=height
            ).reshape(height, width, -1)

            if raster.dtype in [np.uint8, np.uint16]:
                raster = (raster.astype(np.float32) / 255.0).clip(0.0, 1.0)
            elif raster.dtype in [np.float, np.float32]:
                raster = raster.astype(np.float32)
            else:
                raise NotImplementedError()

            input_rasters.append(raster)

        input_raster = np.concatenate(input_rasters, axis=-1)

        imap = (
            dataset_item.map.provide_raster(
                raster_layer_name="imap", x=x, y=y, width=width, height=height
            )
            .reshape(height, width)
            .astype(np.uint16)
        )

        annotations = dataset_item.map.query_intersection(
            vector_layer_name="annotations",
            x=x,
            y=y,
            width=width,
            height=height,
            resolve_objects=True,
        )

        # transform annotations

        for annotation_object_id, annotation_object_data in annotations.items():
            if "boundingBox" in annotation_object_data:
                del annotation_object_data["boundingBox"]

            if "keypoints" in annotation_object_data:
                annotation_object_data["keypoints"] = [
                    {
                        "coordinates": [
                            keypoint_data["coordinates"][0] - x,
                            keypoint_data["coordinates"][1] - y,
                        ]
                    }
                    for keypoint_data in annotation_object_data["keypoints"]
                ]

            if "semanticLabelName" in annotation_object_data:
                semantic_label_name = annotation_object_data["semanticLabelName"]

                if semantic_label_name in self._semantic_labels_mapping:
                    annotation_object_data[
                        "semanticLabelName"
                    ] = self._semantic_labels_mapping[
                        annotation_object_data["semanticLabelName"]
                    ]
                else:
                    # map to first semantic label in list, it should usually be the Soil/Background class
                    logger.error(
                        f"Unexpected semantic label '{semantic_label_name}' "
                        f"in dataset item '{dataset_item.name}' "
                        f"in dataset '{dataset_item.dataset.name}' ({dataset_item.path})."
                    )
                    annotation_object_data["semanticLabelName"] = next(
                        iter(self._semantic_labels_mapping.keys())
                    )

        return input_raster, imap, annotations

    def _geometry_transform_from_data(self, transform_data: typing.Dict):
        if transform_data["type"] == "affine":
            return pi_transform.PiRandomAffineTransform(
                input_width=self._sample_size,
                input_height=self._sample_size,
                output_width=self._input_width,
                output_height=self._input_height,
                **transform_data,
            )

        raise ValueError(
            f"Geometry transform type '{transform_data['type']}' not recognized."
        )

    def _color_transform_from_data(self, transform_data: typing.Dict):
        if transform_data["type"] == "hsv":
            return pi_transform.PiRandomHsvTransform(**transform_data)
        elif transform_data["type"] == "contrast":
            return pi_transform.PiRandomContrastTransform(**transform_data)
        elif transform_data["type"] == "blur":
            return pi_transform.PiRandomBlurTransform(**transform_data)

        raise ValueError(
            f"Color transform type '{transform_data['type']}' not recognized."
        )

    def _input_mean_from_config(self) -> typing.List[float]:
        mean = []
        for input_layer_data in self._config["input_layers"]:
            mean += input_layer_data["mean"]
        return mean

    def _input_std_from_config(self) -> typing.List[float]:
        std = []
        for input_layer_data in self._config["input_layers"]:
            std += input_layer_data["std"]
        return std
