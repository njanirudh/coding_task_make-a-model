#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file is covered by the LICENSE file in the root of this project.

from __future__ import annotations

import typing
import pathlib
import copy

import numpy as np
import cv2

from src.piutils.piutils import pi_io

from .pi_map import PiMap

IMAP_BACKGORUND = 0
IMAP_IGNORE = 1

SAMPLING_INDEX_NAMES = {"instances"}


class PiDataset:
    """A dataset with multiple items.

    Each item has its own pi_map.PiMap.
    """

    def __init__(self, path: pathlib.Path):
        self._path = path

        self._meta = pi_io.read_json(path=self._path / "meta.json")

        self._load_items()
        self._load_sampling_indices()
        self._load_semantics_color_mapping()

    @property
    def path(self) -> pathlib.Path:
        return self._path

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def description(self) -> str:
        return self._meta["description"]

    @property
    def semantic_labels_data(self) -> typing.Dict:
        return copy.deepcopy(self._meta["semanticLabels"])

    @property
    def image_layers_data(self) -> typing.Dict:
        return copy.deepcopy(self._meta["imageLayers"])

    @property
    def semantics_color_mapping(self) -> np.ndarray:
        return self._semantics_color_mapping

    @property
    def items(self) -> typing.Dict[str, PiDatasetItem]:
        return self._items

    def sampling_index(self, sampling_index_name: str) -> CiSamplingIndexType:
        if sampling_index_name not in self._sampling_indices:
            raise ValueError(f"Sampling index does not exist: '{sampling_index_name}'")

        return self._sampling_indices[sampling_index_name]

    def _load_items(self):
        self._items = {
            item_path.name: PiDatasetItem(name=item_path.name, dataset=self)
            for item_path in sorted(list((self._path / "items").iterdir()))
        }

    def _load_sampling_indices(self):
        self._sampling_indices = {
            sampling_index_name: pi_io.read_json(
                path=self._path / "samplingIndices" / f"{sampling_index_name}.json"
            )
            for sampling_index_name in SAMPLING_INDEX_NAMES
        }

    def _load_semantics_color_mapping(self):
        semantics_color_mapping = np.zeros((256, 3), dtype=np.uint8)

        for semantic_label_name, semantic_label_data in self._meta[
            "semanticLabels"
        ].items():
            semantic_label_id = semantic_label_data["imapId"]
            semantic_label_color = semantic_label_data["color"]

            semantics_color_mapping[semantic_label_id] = np.asarray(
                semantic_label_color, dtype=np.uint8
            )

        self._semantics_color_mapping = semantics_color_mapping


class PiDatasetItem:
    def __init__(self, name: str, dataset: CiDatasetType):
        self._name = name
        self._dataset = dataset

        self._map = PiMap(path=self.map_path)

    @property
    def name(self) -> str:
        return self._name

    @property
    def dataset(self) -> CiDatasetType:
        return self._dataset

    @property
    def path(self) -> pathlib.Path:
        return self.dataset.path / "items" / self._name

    @property
    def map_path(self) -> pathlib.Path:
        return self.path / "map"

    @property
    def height(self) -> int:
        return self._map.height

    @property
    def width(self) -> int:
        return self._map.width

    @property
    def map(self) -> PiMap:
        return self._map

    def annotation_object(self, annotation_object_id: str) -> typing.Dict:
        return self._map.vector_object(
            vector_layer_name="annotations", vector_object_id=annotation_object_id
        )
