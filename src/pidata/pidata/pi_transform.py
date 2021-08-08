#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file is covered by the LICENSE file in the root of this project.

from __future__ import annotations

import typing

import cv2
import numpy as np


class PiRandomTransform:
    """A transformation that can act on raster and sparse two-dimensional data."""

    def resample(self, random: np.random.RandomState):
        raise NotImplementedError(f"{type(self)}.resample is not yet implemented.")

    def transform_raster(
            self, raster: np.ndarray, interpolation: str, fill_value: int
    ) -> np.ndarray:
        """
        Args:
            interpolation: One of "nearest", "linear", "cubic", "area".
        """
        raise NotImplementedError(
            f"{type(self)}.transform_raster is not yet implemented."
        )

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Args:
            points: Shape (N, 2,). Order x, y.
        """
        raise NotImplementedError(
            f"{type(self)}.transform_points is not yet implemented."
        )


class PiRandomAffineTransform(PiRandomTransform):
    def __init__(
            self,
            input_width: int,
            input_height: int,
            output_width: int,
            output_height: int,
            flip_x_probability: float,
            flip_y_probability: float,
            rotation_max: float,
            rotation_min: float,
            scaling_x_max: float,
            scaling_x_min: float,
            scaling_y_max: float,
            scaling_y_min: float,
            shearing_x_max: float,
            shearing_x_min: float,
            shearing_y_max: float,
            shearing_y_min: float,
            translation_x_max: float,
            translation_x_min: float,
            translation_y_max: float,
            translation_y_min: float,
            probability: float,
            **kwargs,
    ):
        super().__init__()

        self._input_width = input_width
        self._input_height = input_height
        self._output_width = output_width
        self._output_height = output_height

        self._flip_x_probability = flip_x_probability
        self._flip_y_probability = flip_y_probability
        self._rotation_min = rotation_min
        self._rotation_max = rotation_max
        self._scaling_x_min = scaling_x_min
        self._scaling_x_max = scaling_x_max
        self._scaling_y_min = scaling_y_min
        self._scaling_y_max = scaling_y_max
        self._shearing_x_min = shearing_x_min
        self._shearing_x_max = shearing_x_max
        self._shearing_y_min = shearing_y_min
        self._shearing_y_max = shearing_y_max
        self._translate_x_min = translation_x_min
        self._translate_x_max = translation_x_max
        self._translate_y_min = translation_y_min
        self._translate_y_max = translation_y_max
        self._probability = probability

        self._flip_x = None
        self._flip_y = None
        self._rotation = None
        self._scaling_x = None
        self._scaling_y = None
        self._shearing_x = None
        self._shearing_y = None
        self._translate_x = None
        self._translate_y = None
        self._matrix = None

        self._apply = None

    def resample(self, random: np.random.RandomState):
        self._apply = random.choice(
            [True, False], p=[self._probability, 1.0 - self._probability]
        )

        if not self._apply:
            self._flip_x = None
            self._flip_y = None
            self._rotation = None
            self._scaling_x = None
            self._scaling_y = None
            self._shearing_x = None
            self._shearing_y = None
            self._translate_x = None
            self._translate_y = None
            self._matrix = None
            return

        self._flip_x = random.choice(
            [True, False],
            p=[self._flip_x_probability, 1.0 - self._flip_x_probability],
            replace=False,
        )
        self._flip_y = random.choice(
            [True, False],
            p=[self._flip_y_probability, 1.0 - self._flip_y_probability],
            replace=False,
        )
        self._rotation = random.uniform(self._rotation_min, self._rotation_max)
        self._scaling_x = random.uniform(self._scaling_x_min, self._scaling_x_max)
        self._scaling_y = random.uniform(self._scaling_y_min, self._scaling_y_max)
        self._shearing_x = random.uniform(self._shearing_x_min, self._shearing_x_max)
        self._shearing_y = random.uniform(self._shearing_y_min, self._shearing_y_max)
        self._translate_x = random.uniform(self._translate_x_min, self._translate_x_max)
        self._translate_y = random.uniform(self._translate_y_min, self._translate_y_max)

        # contruct transformation matrix

        translation_1 = np.eye(3, dtype=np.float)
        translation_1[0, 2] = -0.5 * self._input_width
        translation_1[1, 2] = -0.5 * self._input_height

        scaling = np.eye(3, dtype=np.float)
        scaling[0, 0] = self._scaling_x
        scaling[1, 1] = self._scaling_y
        scaling[0, 1] = self._shearing_x
        scaling[1, 0] = self._shearing_y
        scaling[2, 2] = 1.0

        rotation = np.eye(3, dtype=np.float)
        rotation[0, 0] = np.cos(self._rotation)
        rotation[1, 1] = np.cos(self._rotation)
        rotation[0, 1] = -np.sin(self._rotation)
        rotation[1, 0] = np.sin(self._rotation)
        rotation[2, 2] = 1.0

        translation_2 = np.eye(3, dtype=np.float)
        translation_2[0, 2] = self._translate_x
        translation_2[1, 2] = self._translate_y

        translation_3 = np.eye(3, dtype=np.float)
        translation_3[0, 2] = 0.5 * self._output_width
        translation_3[1, 2] = 0.5 * self._output_height

        self._matrix = (
                translation_3 @ translation_2 @ rotation @ scaling @ translation_1
        )

    def transform_raster(
            self,
            raster: np.ndarray,
            interpolation: str,
            fill_value: typing.Union[int, float, np.ndarray],
    ):
        if not self._apply:
            return raster

        interpolation_flag = {
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "area": cv2.INTER_AREA,
        }[interpolation]

        channels = 1 if len(raster.shape) == 2 else raster.shape[2]

        if channels not in [1, 3]:
            # apply on each channel separately
            return np.stack(
                [
                    self.transform_raster(
                        raster=raster[..., channel],
                        interpolation=interpolation,
                        fill_value=fill_value[channel]
                        if isinstance(fill_value, np.ndarray)
                        else fill_value,
                    )
                    for channel in range(channels)
                ],
                axis=-1,
            )

        if isinstance(fill_value, np.ndarray) and fill_value.size == 1:
            fill_value = fill_value.item()
        elif isinstance(fill_value, np.ndarray):
            fill_value = tuple(value.item() for value in fill_value)

        return cv2.warpAffine(
            src=raster,
            M=self._matrix[:2, :],
            dsize=(self._output_width, self._output_height),
            flags=interpolation_flag,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=fill_value,
        )

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if not self._apply:
            return points

        num_points = points.shape[0]
        # using homogeneous coordinates
        points = np.stack(
            [points[:, 0], points[:, 1], np.ones((num_points,), dtype=np.float)],
            axis=-1,
        )
        return ((self._matrix @ points.T).T)[:, :2]


class PiRandomHsvTransform(PiRandomTransform):
    def __init__(
            self,
            hue_min: float,
            hue_max: float,
            saturation_min: float,
            saturation_max: float,
            value_min: float,
            value_max: float,
            probability: float,
            channels: typing.List[int],
            **kwargs,
    ):
        super().__init__()

        if len(channels) != 3:
            raise ValueError("Three channel indices expected.")

        self._hue_min = hue_min
        self._hue_max = hue_max
        self._saturation_min = saturation_min
        self._saturation_max = saturation_max
        self._value_min = value_min
        self._value_max = value_max
        self._probability = probability

        self._channels = channels

        self._hue = None
        self._saturation = None
        self._value = None

        self._apply = None

    def resample(self, random: np.random.RandomState):
        self._apply = random.choice(
            [True, False], p=[self._probability, 1.0 - self._probability]
        )

        if not self._apply:
            self._hue = None
            self._saturation = None
            self._value = None
            return

        self._hue = random.uniform(low=self._hue_min, high=self._hue_max)
        self._saturation = random.uniform(
            low=self._saturation_min, high=self._saturation_max
        )
        self._value = random.uniform(low=self._value_min, high=self._value_max)

    def transform_raster(
            self, raster: np.ndarray, interpolation: str, fill_value: np.ndarray
    ) -> np.ndarray:
        if not self._apply:
            return raster

        rgb = raster[..., self._channels]

        # debug output
        # cv2.imshow("input", rgb[..., ::-1])

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

        # hue
        hsv[..., 0] = np.remainder(
            360.0 * (hsv[..., 0] / 360.0 + 1.0 + self._hue), 360.0
        )

        # saturation
        hsv[..., 1] = np.clip(hsv[..., 1] + self._saturation, 0.0, 1.0)

        # value
        hsv[..., 2] = np.clip(hsv[..., 2] + self._value, 0.0, 1.0)

        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        # debug output
        # cv2.imshow("transformed", rgb[..., ::-1])

        raster[..., self._channels] = rgb

        return raster

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if not self._apply:
            return points

        return points


class PiRandomContrastTransform(PiRandomTransform):
    def __init__(
            self,
            contrast_min: float,
            contrast_max: float,
            probability: float,
            channels: typing.List[int],
            **kwargs,
    ):
        super().__init__()

        self._contrast_min = contrast_min
        self._contrast_max = contrast_max
        self._probability = probability

        self._channels = channels

        self._contrast = None
        self._apply = None

    def resample(self, random: np.random.RandomState):
        self._apply = random.choice(
            [True, False], p=[self._probability, 1.0 - self._probability]
        )

        if not self._apply:
            self._contrast = None
            return

        self._contrast = random.uniform(low=self._contrast_min, high=self._contrast_max)

    def transform_raster(
            self, raster: np.ndarray, interpolation: str, fill_value: np.ndarray
    ) -> np.ndarray:
        if not self._apply:
            return raster

        rgb = raster[..., self._channels]

        # debug output
        # cv2.imshow("input", rgb[..., ::-1])

        mean = np.mean(rgb.reshape(-1, 3), axis=0).reshape(1, 1, 3)
        rgb = np.clip((rgb - mean) * (1.0 + self._contrast) + mean, 0.0, 1.0)

        # debug output
        # cv2.imshow("transformed", rgb[..., ::-1])

        raster[..., self._channels] = rgb

        return raster

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if not self._apply:
            return points

        return points


class PiRandomBlurTransform(PiRandomTransform):
    def __init__(
            self,
            blur_min: float,
            blur_max: float,
            probability: float,
            channels: typing.List[int],
            **kwargs,
    ):
        super().__init__()

        self._blur_min = blur_min
        self._blur_max = blur_max
        self._probability = probability

        self._channels = channels

        self._blur = None

    def resample(self, random: np.random.RandomState):
        self._apply = random.choice(
            [True, False], p=[self._probability, 1.0 - self._probability]
        )

        if not self._apply:
            self._blur = None
            return

        self._blur = random.uniform(low=self._blur_min, high=self._blur_max)

    def transform_raster(
            self, raster: np.ndarray, interpolation: str, fill_value: np.ndarray
    ) -> np.ndarray:
        if not self._apply:
            return raster

        if self._blur == 0.0:
            return raster

        rgb = raster[..., self._channels]

        rgb = cv2.GaussianBlur(rgb, (0, 0), sigmaX=self._blur)

        raster[..., self._channels] = rgb

        return raster

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if not self._apply:
            return points

        return points
