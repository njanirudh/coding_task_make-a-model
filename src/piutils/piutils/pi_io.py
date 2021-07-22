#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file is covered by the LICENSE file in the root of this project.
"""Utility functions for reading/writing files.
"""
from __future__ import annotations

import typing
import pathlib
import json

import cv2
import numpy as np
import yaml


def read_json(path: typing.Union[pathlib.Path, str]) -> typing.Dict:
    with pathlib.Path(path).open("r") as json_file:
        return json.load(json_file)


def read_yaml(path: typing.Union[pathlib.Path, str]) -> typing.Dict:
    with pathlib.Path(path).open("r") as yaml_file:
        return yaml.safe_load(yaml_file)


def write_yaml(data: typing.Dict, path: typing.Union[pathlib.Path, str]):
    with pathlib.Path(path).open("w+") as yaml_file:
        yaml.safe_dump(data, yaml_file)


def read_image(
    path: pathlib.Path, dtype: typing.Optional[numpy.dtype] = None
) -> numpy.ndarray:
    raster = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if raster is None:
        raise ValueError(f"Error reading image: {path}")

    raster = _convert_raster(raster, dtype=dtype)

    return raster


def write_image(raster: numpy.ndarray, path: pathlib.Path, **kwargs) -> None:
    if len(raster.shape) == 3 and raster.shape[2] == 3:
        raster = cv2.cvtColor(raster, cv2.COLOR_RGB2BGR)
    elif len(raster.shape) == 3 and raster.shape[2] == 4:
        raster = cv2.cvtColor(raster, cv2.COLOR_RGBA2BGRA)

    success = cv2.imwrite(str(path), raster, **kwargs)

    if not success:
        raise Exception(f"Error writing image: {path}")


def read_binary(path: pathlib.Path) -> bytes:
    with path.open("rb") as binary_file:
        return binary_file.read()


def decode_image(binary: bytes, dtype: typing.Optional[numpy.dtype] = None):
    binary_array = np.fromstring(binary, dtype=np.uint8)
    raster = cv2.imdecode(binary_array, cv2.IMREAD_UNCHANGED)

    if raster is None:
        raise ValueError(f"Error decoding image.")

    raster = _convert_raster(raster, dtype=dtype)

    return raster


def _convert_raster(raster: np.ndarray, dtype: typing.Optional[numpy.dtype] = None):
    if len(raster.shape) == 3 and raster.shape[2] == 3:
        raster = cv2.cvtColor(raster, cv2.COLOR_BGR2RGB)
    elif len(raster.shape) == 3 and raster.shape[2] == 4:
        raster = cv2.cvtColor(raster, cv2.COLOR_BGRA2RGBA)

    if dtype is not None:
        raster = raster.astype(dtype)

    return raster
