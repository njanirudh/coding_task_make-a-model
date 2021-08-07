import cv2
import torch
import numpy as np

from src.pidata.pidata import pi_parser
from src.utils.custom_config import custom_parser_config
from src.piutils.piutils import pi_drawing

from src.piutils.piutils import pi_log
from src.piutils.piutils import pi_io
from src.piutils.piutils import pi_drawing
from src.pidata.pidata import pi_parser

logger = pi_log.get_logger(__name__)

from seg_trainer import SegmentationModule


import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TKAgg', warn=False, force=True)


def inference(input_img: torch.tensor, model_path: str) -> list:
    seg_inference = SegmentationModule(config_data=custom_parser_config,
                                       train_mode=False,
                                       batch_size=5,
                                       epochs=150,
                                       gpu=1)
    seg_inference.model.eval()
    seg_inference.load_state_dict(torch.load(model_path), strict=False)

    with torch.no_grad():
        output_vals = seg_inference(input_img)

    return output_vals


if __name__ == "__main__":

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
            output_width=train_data_parser.output_width,  # position of network output wrt to input, see pidata.pi_parser.__init__()
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
        if sample_index != 18:
            continue

        print(input_tensor.shape)
        input_img = torch.unsqueeze(torch.from_numpy(input_tensor), 0)
        print(input_img.shape)


        # Model Output :
        #       ['boxes', 'labels', 'scores', 'masks']
        result_dict = inference(input_img, MODEL_CHKP_PATH)[0]
        print(type(result_dict), result_dict.keys())
        print(type(result_dict['masks']))

        drawing_input = pi_drawing.draw_input(
            input_tensor=input_tensor,
            **drawing_kwargs
        )

        print(drawing_input[..., ::-1].shape)
        plt.figure(100)
        plt.imshow(drawing_input)

        # cv2.imshow(
        #     "input",
        #     drawing_input[..., ::-1],  # RGB to BGR
        # )

        #--------------------------------------------
        # -------------Semantic Map------------------
        #--------------------------------------------
        if "semantics" in target_dict:
            drawing_semantic_labels = (
                pi_drawing.color_semantic_labels(  # feel free to use in you own code
                    semantics_tensor=target_dict["semantics"],
                    **drawing_kwargs
                )
            )

            plt.figure(200)
            plt.imshow(drawing_semantic_labels)

            # cv2.imshow(
            #     "semantic_labels",
            #     drawing_semantic_labels[..., ::-1],  # RGB to BGR
            # )

        #--------------------------------------------
        # ---------------Result Viz------------------
        #--------------------------------------------
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
                    **drawing_kwargs,
                )
            )

            plt.figure(300)
            plt.imshow(drawing_instances)
            # cv2.imshow(
            #     "instances",
            #     drawing_instances[..., ::-1],  # RGB to BGR
            # )

        plt.show()
        # cv2.waitKey()
        break
