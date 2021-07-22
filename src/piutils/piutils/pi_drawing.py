#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file is covered by the LICENSE file in the root of this project.
"""A number of drawing functions to visualize the parsed network input/targets.
"""

from __future__ import annotations

import typing

import cv2
import numpy as np


def draw_input(
    input_tensor: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    output_width: int,
    output_height: int,
    output_offset_x: int = 0,
    output_offset_y: int = 0,
    output_stride_x: int = 1,
    output_stride_y: int = 1,
    scale_factor: float = 1.0,
    **kwargs,
):
    """Make a human-viewable image of the parsed network input.

    Args:
        input_tensor:
            A numpy.ndarray of shape
               (batch_size, channels, height, width) or
               (channels, height, width) and
            dtype numpy.float32.
        mean, std:
            A numpy.ndarray of dtype numpy.float32.
            Used to reverse input data normalization.
        output_*:
            Network output relative to the network input.
            Used to crop/scale the drawings.
            Also see pidata.pi_parser.PiParser.__init__().
        scale_factor:
            Set to scale the drawing.

    Returns:
        A numpy.ndarray of dtype numpy.uint8.
    """
    return np.concatenate(
        _make_list_of_input_drawings(
            input_tensor=input_tensor,
            mean=mean,
            std=std,
            output_width=output_width,
            output_height=output_height,
            output_offset_x=output_offset_x,
            output_offset_y=output_offset_y,
            output_stride_x=output_stride_x,
            output_stride_y=output_stride_y,
            scale_factor=scale_factor,
        ),
        axis=1,
    )


def color_semantic_labels(
    semantics_tensor: np.ndarray,
    semantic_labels: typing.Dict,
    output_width: int,
    output_height: int,
    output_offset_x: int = 0,
    output_offset_y: int = 0,
    output_stride_x: int = 1,
    output_stride_y: int = 1,
    scale_factor: float = 1.0,
    **kwargs,
) -> np.ndarray:
    """Make a human-viewable image of the parsed target for semantic segmentation network.

    Args:
        semantics_tensor:
            A numpy.ndarray of shape
               (batch_size, height, width) or
               (height, width) and
            dtype numpy.int64 with pixel-wise label IDs.
            See pi_parser.PiParser.__getitem__().
        semantic_labels:
            A nested dictionary of semantic labels with keys 'color' and 'index'.
            As returned by pidata.pi_parser.PiParser.semantic_labels.
        *:
            See draw_input().

    Returns:
        A numpy.ndarray of dtype numpy.uint8.
    """

    if len(semantics_tensor.shape) == 2:
        batch_size = 1
        semantics_tensor = semantics_tensor[np.newaxis]
    elif len(semantics_tensor.shape) == 3:
        batch_size = semantics_tensor.shape[0]
    else:
        raise ValueError(
            f"Unexpected shape of 'semantics_tensor': {semantics_tensor.shape}"
        )

    color_mapping = _make_color_mapping(semantic_labels=semantic_labels)

    drawing_width = np.ceil(scale_factor * output_width).astype(np.int).item()
    drawing_height = np.ceil(scale_factor * output_height).astype(np.int).item()

    drawings = [
        cv2.resize(
            np.take(color_mapping, semantics_tensor[slice_index], axis=0),
            (drawing_width, drawing_height),
            interpolation=cv2.INTER_NEAREST,
        )
        for slice_index in range(batch_size)
    ]

    return np.concatenate(drawings, axis=1)


def draw_instances(
    input_tensor: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    labels: np.ndarray,
    semantic_labels: typing.Dict,
    output_width: int,
    output_height: int,
    output_offset_x: int = 0,
    output_offset_y: int = 0,
    output_stride_x: int = 1,
    output_stride_y: int = 1,
    scale_factor: float = 1.0,
    boxes: typing.Optional[typing.Union[np.ndarray, typing.List[np.ndarray]]] = None,
    keypoints: typing.Optional[
        typing.Union[np.ndarray, typing.List[np.ndarray]]
    ] = None,
    masks: typing.Optional[typing.Union[np.ndarray, typing.List[np.ndarray]]] = None,
    scores: typing.Optional[typing.Union[np.ndarray, typing.List[np.ndarray]]] = None,
    **kwargs,
) -> np.ndarray:
    """Make a human-viewable image of the parsed target for an instance segmentation or object detection network.

    Args:
        labels, boxes, keypoints, masks, scores:
            A numpy.ndarray or a list of numpy.ndarrays (for batched input).
            Also see pidata.pi_parser.PiParser.__getitem__().
        *:
            See draw_input() and colored_semantics().

    Returns:
        A numpy.ndarray of dtype numpy.uint8.
    """

    if isinstance(boxes, np.ndarray) and len(boxes.shape) == 2:
        boxes = [boxes]

    if isinstance(keypoints, np.ndarray) and len(keypoints.shape) == 3:
        keypoints = [keypoints]

    if isinstance(masks, np.ndarray) and len(masks.shape) == 3:
        masks = [masks]

    if isinstance(labels, np.ndarray) and len(labels.shape) == 1:
        labels = [labels]

    if isinstance(scores, np.ndarray) and len(scores.shape) == 1:
        scores = [scores]

    boxes = None if boxes is None or not len(boxes) else boxes
    keypoints = None if keypoints is None or not len(keypoints) else keypoints
    masks = None if masks is None or not len(masks) else masks
    labels = None if labels is None or not len(labels) else labels
    scores = None if scores is None or not len(scores) else scores

    backgrounds = _make_list_of_input_drawings(
        input_tensor=input_tensor,
        mean=mean,
        std=std,
        output_width=output_width,
        output_height=output_height,
        output_offset_x=output_offset_x,
        output_offset_y=output_offset_y,
        output_stride_x=output_stride_x,
        output_stride_y=output_stride_y,
        scale_factor=scale_factor,
    )

    batch_size = len(backgrounds)

    color_mapping = _make_color_mapping(semantic_labels=semantic_labels)
    name_mapping = list(semantic_labels.keys())

    drawing_width = np.ceil(scale_factor * output_width).astype(np.int).item()
    drawing_height = np.ceil(scale_factor * output_height).astype(np.int).item()

    actual_scale_factor_x = drawing_width / output_width * output_stride_x
    actual_scale_factor_y = drawing_height / output_height * output_stride_y

    drawings = []

    for slice_index in range(batch_size):

        if boxes is not None:
            boxes_slice = boxes[slice_index]
            boxes_slice = (
                boxes_slice
                * np.asarray(
                    [
                        actual_scale_factor_x,
                        actual_scale_factor_y,
                        actual_scale_factor_x,
                        actual_scale_factor_y,
                    ],
                    dtype=np.float32,
                ).reshape(1, 4)
            )
            boxes_slice_int = np.round(boxes_slice).astype(np.int)

        if keypoints is not None:
            keypoints_slice = keypoints[slice_index]

            keypoints_slice = (
                keypoints_slice
                * np.asarray(
                    [actual_scale_factor_x, actual_scale_factor_y, 1.0],
                    dtype=np.float32,
                ).reshape(1, 1, 3)
            )
            keypoints_slice_int = np.round(keypoints_slice).astype(np.int)

        if masks is not None:
            masks_slice = masks[slice_index]

        if labels is not None:
            labels_slice = labels[slice_index]

        if scores is not None:
            scores_slice = scores[slice_index]

        drawing = backgrounds[slice_index]

        if masks is not None and labels is not None:
            for mask, label in zip(masks_slice, labels_slice):
                color = color_mapping[label].reshape(1, 1, 3)

                mask = cv2.resize(
                    mask,
                    (drawing_width, drawing_height),
                    interpolation=cv2.INTER_NEAREST,
                )

                drawing = np.where(mask[..., np.newaxis] > 0, color, drawing)

                contour = _contour(mask, thickness=1)

                drawing = np.where(
                    contour[..., np.newaxis] > 0,
                    np.asarray([255, 255, 255], dtype=np.uint8).reshape(1, 1, 3),
                    drawing,
                )

        if boxes is not None and labels is not None:
            for box_int, label in zip(boxes_slice_int, labels_slice):

                color = (
                    color_mapping[label].tolist() if masks is None else (255, 255, 255)
                )

                drawing = cv2.rectangle(
                    drawing,
                    (
                        box_int[0],
                        box_int[1],
                        box_int[2] - box_int[0],
                        box_int[3] - box_int[1],
                    ),
                    color=color,
                    thickness=1,
                )

        if boxes is not None and keypoints is not None and labels is not None:
            for keypoint_int, box_int, label in zip(
                keypoints_slice_int, boxes_slice_int, labels_slice
            ):
                if not keypoint_int[0, 2].item():
                    # not visible
                    continue

                color = (
                    color_mapping[label].tolist() if masks is None else (255, 255, 255)
                )

                drawing = cv2.line(
                    drawing,
                    (
                        keypoint_int[0, 0]
                        + 5 * (-1 if box_int[0] < keypoint_int[0, 0] else 1),
                        keypoint_int[0, 1],
                    ),
                    (box_int[0], keypoint_int[0, 1]),
                    thickness=1,
                    color=color,
                )

                drawing = cv2.line(
                    drawing,
                    (box_int[0], keypoint_int[0, 1]),
                    (box_int[0], box_int[1]),
                    thickness=1,
                    color=color,
                )

        if keypoints is not None and labels is not None:
            for keypoint_int, label in zip(keypoints_slice_int, labels_slice):
                if not keypoint_int[0, 2].item():
                    # not visible
                    continue

                color = (
                    color_mapping[label].tolist() if masks is None else (255, 255, 255)
                )

                drawing = cv2.circle(
                    drawing,
                    (
                        keypoint_int[0, 0],
                        keypoint_int[0, 1],
                    ),
                    radius=5,
                    color=color,
                    lineType=cv2.LINE_AA,
                    thickness=1,
                )

        if boxes is not None and scores is not None and labels is not None:
            for box_int, score, label in zip(
                boxes_slice_int, scores_slice, labels_slice
            ):
                semantic_label_name = name_mapping[label]
                color = (
                    color_mapping[label].tolist() if masks is None else (255, 255, 255)
                )
                drawing = cv2.putText(
                    drawing,
                    f"{semantic_label_name.split('.')[-1]}: {score:.3f}",
                    (box_int[0], box_int[1] - 5),
                    color=color,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    lineType=cv2.LINE_AA,
                )

        drawings.append(drawing)

    return np.concatenate(drawings, axis=1)


def _make_list_of_input_drawings(
    input_tensor: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    output_width: int,
    output_height: int,
    output_offset_x: int,
    output_offset_y: int,
    output_stride_x: int,
    output_stride_y: int,
    scale_factor: float,
) -> typing.List[np.ndarray]:

    if len(input_tensor.shape) == 3:
        batch_size = 1
        input_tensor = input_tensor[np.newaxis]
    elif len(input_tensor.shape) == 4:
        batch_size = input_tensor.shape[0]
    else:
        raise ValueError(f"Unexpected shape of 'input_tensor': {input_tensor.shape}")

    input_tensor = input_tensor.transpose((0, 2, 3, 1))  # channels last
    channels = input_tensor.shape[-1]

    drawing_width = np.ceil(scale_factor * output_width).astype(np.int).item()
    drawing_height = np.ceil(scale_factor * output_height).astype(np.int).item()

    if std is not None and mean is not None:
        input_tensor = input_tensor * std.reshape(1, 1, 1, channels) + mean.reshape(
            1, 1, 1, channels
        )

    drawings = (
        np.ascontiguousarray(255.0 * input_tensor[..., :3])
        .clip(0.0, 255.0)
        .astype(np.uint8)
    )

    return [
        cv2.resize(
            drawing[
                output_offset_y : (output_offset_y + output_height),
                output_offset_x : (output_offset_x + output_width),
            ],
            (drawing_width, drawing_height),
            interpolation=cv2.INTER_NEAREST,
        )
        for drawing in drawings
    ]


def _make_color_mapping(semantic_labels: typing.Dict) -> np.ndarray:
    color_mapping = np.zeros((len(semantic_labels) + 1, 3), dtype=np.uint8)

    for semantic_label_data in semantic_labels.values():
        color_mapping[semantic_label_data["index"]] = np.asarray(
            semantic_label_data["color"], dtype=np.uint8
        )

    ignore_index = len(semantic_labels)

    # draw ignored areas in light gray
    color_mapping[ignore_index] = np.asarray([200, 200, 200], dtype=np.uint8)

    return color_mapping


def _contour(mask: np.ndarray, thickness: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (2 * thickness + 1, 2 * thickness + 1)
    )
    return np.where(
        mask != cv2.erode(mask, kernel),
        mask,
        0,
    )
