import numpy as np
import torch

from src.pidata.pidata import pi_parser
from src.piutils.piutils import pi_drawing
from src.piutils.piutils import pi_log
from src.utils.custom_config import custom_parser_config

logger = pi_log.get_logger(__name__)

from seg_trainer import SegmentationModule

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TKAgg', warn=False, force=True)


# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


def inference(input_img: torch.Tensor, model_path: str) -> torch.Tensor:
    seg_inference = SegmentationModule(config_data=custom_parser_config,
                                       train_mode=False,
                                       batch_size=5,
                                       epochs=150,
                                       gpu=1)
    seg_inference.model.eval()
    # seg_inference.load_state_dict(torch.load(model_path), strict=False)

    with torch.no_grad():
        output_vals = seg_inference(input_img)

    return output_vals


if __name__ == "__main__":

    train_data_parser = pi_parser.PiParser(
        config=custom_parser_config,
        split_name="val",
        num_samples=10,  # number of samples to be drawn, set to a multiple of the batch size
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

    MODEL_CHKP_PATH = "/home/anirudh/NJ/Interview/Pheno-Inspect/git_proj/" \
                      "coding_task_make-a-model/src/lightning_logs/version_1/" \
                      "checkpoints/epoch=92-step=184.ckpt"

    # Target Values :
    #       ['semantics', 'boxes', 'labels', 'area', 'iscrowd', 'masks', 'keypoints', 'image_id']
    for sample_index, (input_tensor, target_dict) in enumerate(train_data_parser):
        if sample_index != 7:
            continue

        print("Input Shape : ", input_tensor.shape)
        input_img = torch.unsqueeze(torch.from_numpy(input_tensor), 0)
        print("Input Shape Unsqueezed : ", input_img.shape)

        # Model Output :
        #       ['boxes', 'labels', 'scores', 'masks']
        result_dict = inference(input_img, MODEL_CHKP_PATH)[0]
        print("Result Dict : ", type(result_dict), result_dict.keys())
        print("boxes : ", result_dict['boxes'].shape, type(result_dict['boxes']))
        print("labels : ", result_dict['labels'].shape, type(result_dict['labels']))
        print("scores : ", result_dict['scores'].shape, type(result_dict['scores']))
        print("masks : ", result_dict['masks'].shape, type(result_dict['masks']))

        drawing_input = pi_drawing.draw_input(
            input_tensor=input_tensor,
            **drawing_kwargs
        )

        plt.figure(100)
        plt.imshow(drawing_input)


        # --------------------------------------------
        # -------------Semantic Map------------------
        # --------------------------------------------
        if "semantics" in target_dict:
            drawing_semantic_labels = (
                pi_drawing.color_semantic_labels(  # feel free to use in you own code
                    semantics_tensor=target_dict["semantics"],
                    **drawing_kwargs
                )
            )

            print("Semantic Map : ", target_dict["semantics"].shape,
                  np.unique(target_dict["semantics"]))

            plt.figure(200)
            plt.imshow(drawing_semantic_labels)


        # --------------------------------------------
        # ---------------Result Viz------------------
        # --------------------------------------------
        if (
                "boxes" in result_dict
                or "keypoints" in result_dict
                or "masks" in result_dict
        ) and "labels" in result_dict:
            drawing_instances = (
                pi_drawing.draw_instances(  # feel free to use in your own code
                    input_tensor=input_tensor,
                    boxes=np.array(result_dict["boxes"]) if "boxes" in result_dict else None,
                    keypoints=(
                        np.array(result_dict["keypoints"]) if "keypoints" in result_dict else None
                    ),
                    masks=np.array(result_dict["masks"]) if "masks" in result_dict else None,
                    labels=np.array(result_dict["labels"]),
                    **drawing_kwargs
                )
            )

            plt.figure(300)
            plt.imshow(drawing_instances)

        plt.show()
        # cv2.waitKey()
        break
