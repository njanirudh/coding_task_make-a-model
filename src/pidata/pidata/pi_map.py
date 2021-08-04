#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file is covered by the LICENSE file in the root of this project.

from __future__ import annotations

import itertools
import math
import pathlib
import typing

import numpy as np
import rtree

from src.piutils.piutils import pi_io
from src.piutils.piutils import pi_log

logger = pi_log.get_logger(__name__)


class PiMap:
    """A tiled map with raster and vector layers."""

    def __init__(self, path: pathlib.Path):
        if not path.is_dir():
            raise ValueError("Not a directory: '{path}'")

        self._path = path

        self._meta = pi_io.read_json(path=self._path / "meta.json")

        self._load_spatial_indices()

    @property
    def height(self) -> int:
        return self._meta["height"]

    @property
    def width(self) -> int:
        return self._meta["width"]

    @property
    def path(self) -> pathlib.Path:
        return self._path

    @property
    def tile_grids_data(self) -> typing.List[typing.Dict]:
        return self._meta["tileGrids"]

    def spatial_index(self, vector_layer_name: str) -> rtree.index.Rtree:
        return self._spatial_indices[vector_layer_name]

    def raster_layer(self, raster_layer_name: str) -> typing.Dict:
        return self._meta["rasterLayers"][raster_layer_name]

    def find_coarsest_tile_grid(self, scale_denominator: int) -> typing.Dict:
        """Find the tile grid with the greatest scale denominator less than or equal the given one.

        If no such tile grid exists return the tile grid with the smallest scale denominator.

        Raises:
            ValueError: If not tile grids exist at all.
        """
        coarsest = None
        smallest = None

        for tile_grid_data in self._meta["tileGrids"]:
            if tile_grid_data["scaleDenominator"] <= scale_denominator and (
                coarsest is None
                or tile_grid_data["scaleDenominator"] > coarsest["scaleDenominator"]
            ):
                coarsest = tile_grid_data

            if (
                smallest is None
                or tile_grid_data["scaleDenominator"] < smallest["scaleDenominator"]
            ):
                smallest = tile_grid_data

        if coarsest is not None:
            return coarsest

        if smallest is not None:
            return smallest

        raise ValueError(f"No tile grids exist for this map: {self._path}")

    def provide_raster(
        self,
        raster_layer_name: str,
        x: int,
        y: int,
        width: int,
        height: int,
        scale_denominator: int = 1,
    ):
        if scale_denominator != 1:
            raise ValueError("A scale denominator different from 1 is not supported.")

        tile_grid_data = self.find_coarsest_tile_grid(
            scale_denominator=scale_denominator
        )

        tile_width = tile_grid_data["tileWidth"]
        tile_height = tile_grid_data["tileHeight"]

        raster_layer_data = self.raster_layer(raster_layer_name=raster_layer_name)

        num_channels = raster_layer_data["numChannels"]
        dtype = raster_layer_data["dtype"]
        default_value = raster_layer_data["defaultValue"]
        file_format = raster_layer_data["fileFormat"]

        max_tile_x = math.ceil(self.width / tile_width) - 1
        max_tile_y = math.ceil(self.height / tile_height) - 1

        raster_min_tile_x = math.floor(x / tile_width)
        raster_max_tile_x = math.floor((x + width) / tile_width)
        raster_min_tile_y = math.floor(y / tile_height)
        raster_max_tile_y = math.floor((y + height) / tile_height)

        tile_x_indices = range(raster_min_tile_x, raster_max_tile_x + 1)
        tile_y_indices = range(raster_min_tile_y, raster_max_tile_y + 1)

        num_tiles_x = len(tile_x_indices)
        num_tiles_y = len(tile_y_indices)

        tile_indices = itertools.product(tile_x_indices, tile_y_indices)

        if num_channels == 1:
            tiles_shape = (num_tiles_y * tile_height, num_tiles_x * tile_width)
        else:
            tiles_shape = (
                num_tiles_y * tile_height,
                num_tiles_x * tile_width,
                num_channels,
            )

        raster = np.full(
            tiles_shape,
            default_value,
            dtype=dtype,
        )

        for tile_x, tile_y in tile_indices:
            if tile_x < 0 or tile_x > max_tile_x or tile_y < 0 or tile_y > max_tile_y:
                continue

            raster_left = (tile_x - raster_min_tile_x) * tile_width
            raster_upper = (tile_y - raster_min_tile_y) * tile_height

            raster[
                raster_upper : raster_upper + tile_height,
                raster_left : raster_left + tile_width,
            ] = self._read_tile(
                raster_layer_name=raster_layer_name,
                tile_x=tile_x,
                tile_y=tile_y,
                tile_width=tile_width,
                tile_height=tile_height,
                scale_denominator=scale_denominator,
                num_channels=num_channels,
                dtype=dtype,
                file_format=file_format,
            )

        tiles_left = tile_width * raster_min_tile_x
        tiles_upper = tile_height * raster_min_tile_y

        raster = raster[
            y - tiles_upper : y + height - tiles_upper,
            x - tiles_left : x + width - tiles_left,
        ]

        if raster.shape[0] != height or raster.shape[1] != width:
            raise RuntimeError(
                f"Size mismatch. Expected: ({height}, {width}). Got: ({raster.shape[:2]}) "
            )

        return raster

    def _read_tile(
        self,
        raster_layer_name: str,
        tile_x: int,
        tile_y: int,
        tile_width: int,
        tile_height: int,
        scale_denominator: int,
        num_channels: int,
        dtype: str,
        file_format: str,
    ) -> np.ndarray:

        tile_path = self._tile_path(
            raster_layer_name=raster_layer_name,
            tile_x=tile_x,
            tile_y=tile_y,
            scale_denominator=scale_denominator,
            file_format=file_format,
        )

        if not tile_path.exists():
            raise FileNotFoundError(f"Tile not found: '{tile_path}'")

        raster = pi_io.read_image(path=tile_path, dtype=dtype)

        if raster.shape[0] != tile_height or raster.shape[1] != tile_width:
            raise ValueError(
                f"Tile size mismatch. Expected: ({tile_height}, {tile_width}), Got: {raster.shape}"
            )

        if num_channels == 1 and len(raster.shape) != 2:
            raise ValueError(f"Single channel raster expected: '{tile_path}'")

        if num_channels != 1 and (
            len(raster.shape) != 3 or raster.shape[2] != num_channels
        ):
            raise ValueError(
                f"{num_channels} channels raster expected: '{tile_path}' Got: {raster.shape}"
            )

        return raster

    def query_intersection(
        self,
        vector_layer_name: str,
        x: float,
        y: float,
        width: float,
        height: float,
        resolve_objects: bool = False,
    ):
        if vector_layer_name not in self._meta["vectorLayers"]:
            raise ValueError("Vector layer '{vector_layer_name}' does not exist.")

        spatial_index = self._spatial_indices[vector_layer_name]

        left = x
        right = x + width
        upper = y
        lower = y + height

        vector_object_ids = [
            item.object
            for item in spatial_index.intersection(
                (left, right, upper, lower), objects=True
            )
        ]

        if resolve_objects:
            return {
                id_: self.vector_object(
                    vector_layer_name=vector_layer_name, vector_object_id=id_
                )
                for id_ in vector_object_ids
            }

        return vector_object_ids

    def vector_object(self, vector_layer_name: str, vector_object_id: str):
        vector_object_path = self._vector_object_path(
            vector_layer_name=vector_layer_name, vector_object_id=vector_object_id
        )

        if not vector_object_path.exists():
            raise FileNotFoundError("File does not exist: '{vector_object_path}'")

        vector_object_data = pi_io.read_json(path=vector_object_path)

        return vector_object_data

    def _tile_path(
        self,
        raster_layer_name: str,
        tile_x: int,
        tile_y: int,
        scale_denominator: int,
        file_format: str,
    ) -> pathlib.Path:
        return (
            self._path
            / "tileLayers"
            / raster_layer_name
            / "tiles"
            / f"{tile_x}-{tile_y}-{scale_denominator}{file_format}"
        )

    def _load_spatial_indices(self) -> typing.Dict[str, rtree.index.Rtree]:
        self._spatial_indices = {
            vector_layer_name: self._create_spatial_index(
                vector_layer_name=vector_layer_name, from_existing=True
            )
            for vector_layer_name in self._meta["vectorLayers"].keys()
        }

    def _create_spatial_index(
        self, vector_layer_name: str, from_existing: bool
    ) -> rtree.index.Rtree:
        spatial_index_path = self._spatial_index_path(
            vector_layer_name=vector_layer_name
        )

        if not from_existing and spatial_index_path.exists():
            raise FileExistsError(f"File exists: '{spatial_index_path}'")

        if from_existing and not (spatial_index_path.with_suffix(".dat")).exists():
            raise FileNotFoundError(
                f"File does not exist: '{spatial_index_path.with_suffix('.dat')}'"
            )

        if from_existing and not (spatial_index_path.with_suffix(".idx")).exists():
            raise FileNotFoundError(
                f"File does not exist: '{spatial_index_path.with_suffix('.idx')}'"
            )

        if not from_existing:
            spatial_index_path.parent.mkdir(parents=True)

        return rtree.index.Rtree(
            str(spatial_index_path),
            interleaved=False,  # if False coordinate order in spatial index is left/min_x, right/max_x, upper/min_y, lower/max_y
        )

    def _spatial_index_path(self, vector_layer_name: str) -> pathlib.Path:
        return self._path / "vectorLayers" / vector_layer_name / "spatialIndex/rtree"

    def _vector_object_path(
        self, vector_layer_name: str, vector_object_id: str
    ) -> pathlib.Path:
        return (
            self._path
            / f"vectorLayers"
            / vector_layer_name
            / "objects"
            / f"{vector_object_id}.json"
        )

    def _tile_indices(
        self, tile_grid_data: typing.Dict
    ) -> typing.Generator[typing.Tuple[int, int, int, int, int], None, None]:
        tile_width = tile_grid_data["tileWidth"]
        tile_height = tile_grid_data["tileHeight"]
        scale_denominator = tile_grid_data["scaleDenominator"]

        max_tile_x = math.ceil(self.width / tile_width) - 1
        max_tile_y = math.ceil(self.height / tile_height) - 1

        for tile_x, tile_y in itertools.product(
            range(max_tile_x + 1),
            range(max_tile_y + 1),
        ):
            yield tile_x, tile_y, tile_width, tile_height, scale_denominator
