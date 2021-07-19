import cv2
import numpy as np


def color_semantic_labels(
    semantic_labels_tensor: np.ndarray, semantic_labels: typing.Dict
) -> np.ndarray:
    if len(semantic_labels_tensor.shape) == 2:
        batch_size = 1
        semantic_labels_tensor = semantic_labels_tensor[np.newaxis]
    elif len(semantic_labels_tensor.shape) == 3:
        batch_size = semantic_labels_tensor.shape[0]
    else:
        raise ValueError(
            f"Unexpected shape of 'semantic_labels_tensor': {semantic_labels_tensor.shape}"
        )

    print(semantic_labels)

    drawings = []

    for slice_index in range(batch_size):
        pass
