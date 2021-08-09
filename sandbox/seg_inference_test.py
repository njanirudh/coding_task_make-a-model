import random
from typing import List

import cv2
import numpy as np
import torch
from src.utils.custom_config import custom_parser_config

from src.pidata.pidata import pi_parser
from src.piutils.piutils import pi_log

logger = pi_log.get_logger(__name__)

from seg_trainer import SegmentationModule

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TKAgg', warn=False, force=True)

COLORS = np.random.uniform(0, 255, size=(10, 3))


def get_outputs(image, threshold: 0.5, label_list: List):
    model = SegmentationModule(config_data=custom_parser_config,
                               train_mode=False,
                               batch_size=5,
                               epochs=150,
                               gpu=1)
    model.model.eval()

    with torch.no_grad():
        # forward pass of the image through the modle
        outputs = model(image)

    # get all the scores
    scores = list(outputs[0]['scores'].detach().cpu().numpy())
    # index of those scores which are above a certain threshold
    thresholded_preds_inidices = [scores.index(i) for i in scores if i > threshold]
    thresholded_preds_count = len(thresholded_preds_inidices)
    # get the masks
    masks = (outputs[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
    # discard masks for objects which are below threshold
    masks = masks[:thresholded_preds_count]
    # get the bounding boxes, in (x1, y1), (x2, y2) format
    boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))]
             for i in outputs[0]['boxes'].detach().cpu()]
    # discard bounding boxes below threshold value
    boxes = boxes[:thresholded_preds_count]
    # get the classes labels
    labels = [label_list[i] for i in outputs[0]['labels']]
    return masks, boxes, labels


def draw_segmentation_map(image, masks, boxes, labels):
    alpha = 1
    beta = 0.6  # transparency for the segmentation map
    gamma = 0  # scalar added to each sum
    for i in range(len(masks)):
        red_map = np.zeros_like(masks[i]).astype(np.uint8)
        green_map = np.zeros_like(masks[i]).astype(np.uint8)
        blue_map = np.zeros_like(masks[i]).astype(np.uint8)
        # apply a randon color mask to each object
        color = COLORS[random.randrange(0, len(COLORS))]
        red_map[masks[i] == 1], green_map[masks[i] == 1], blue_map[masks[i] == 1] = color
        # combine all the masks into a single image
        segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
        print("Segmentation Map : ", segmentation_map.shape)
        # convert the original PIL image into NumPy format
        image = np.array(image)
        # convert from RGN to OpenCV BGR format
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # apply mask on the image
        print("BGR Image Size : ", image.shape)
        cv2.addWeighted(image, alpha, segmentation_map, beta, gamma, dtype=1)
        # print(segmentation_map.shape)
        # draw the bounding boxes around the objects
        cv2.rectangle(image, boxes[i][0], boxes[i][1], color=color,
                      thickness=2)
        # put the label text above the objects
        cv2.putText(image, labels[i], (boxes[i][0][0], boxes[i][0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color,
                    thickness=2, lineType=cv2.LINE_AA)

    return image


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

    class_names = ['A', 'B', 'C', 'D']

    train_data_parser = pi_parser.PiParser(
        config=custom_parser_config,
        split_name="val",
        num_samples=100,  # number of samples to be drawn, set to a multiple of the batch size
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

        print("Input Img : ", input_tensor.shape)
        input_img = np.swapaxes(input_tensor, 0, 2)
        print("Input Swapped : ", input_img.shape)

        plt.figure(100)
        plt.imshow(input_img)

        input_img_batched = torch.unsqueeze(torch.from_numpy(input_tensor), 0)
        # print(input_img_batched.shape)

        label_list = ['A', 'B', 'C', 'D']
        m, b, l = get_outputs(input_img_batched, 0.5, label_list)

        img = draw_segmentation_map(input_img, m, b, l)

        plt.figure(200)
        plt.imshow(img)

        plt.show()
        break
