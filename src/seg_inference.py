import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

matplotlib.use('TKAgg', warn=False, force=True)

from seg_trainer import SegmentationModule
from src.pidata.pidata import pi_parser
from src.piutils.piutils import pi_drawing
from src.piutils.piutils import pi_log
from src.utils.custom_config import custom_parser_config
from src.utils.non_max_suppression import non_max_suppression_fast

logger = pi_log.get_logger(__name__)


def run_inference(input_img: torch.Tensor, model_path: str) -> torch.Tensor:
    """
    Runs inference on a single image or a batch of images.
    Args:
        input_img: Input single/batch or RGB images [B,C,W,H]
        model_path: Path to model checkpoint (*.ckpt)

    Returns: Results from Mask-RCNN
            [{'boxes', 'labels', 'scores', 'masks'},...,B]

    """
    seg_inference = SegmentationModule.load_from_checkpoint(checkpoint_path=model_path,
                                                            config_data=custom_parser_config,
                                                            train_mode=False,
                                                            batch_size=5,
                                                            epochs=150,
                                                            gpu=1,
                                                            strict=False)
    seg_inference.eval()

    with torch.no_grad():
        output_vals = seg_inference(input_img)

    return output_vals


if __name__ == "__main__":

    # dataset test parser
    train_data_parser = pi_parser.PiParser(
        config=custom_parser_config,
        split_name="test",
        num_samples=1500,  # number of samples to be drawn, set to a multiple of the batch size
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

    MODEL_CHKP_PATH = "/home/anirudh/NJ/Interview/Pheno-Inspect/git_proj/coding_task_make-a-model/" \
                      "src/lightning_logs/version_9/checkpoints/epoch=0-step=98.ckpt"

    # Target Values :
    #       ['semantics', 'boxes', 'labels', 'area', 'iscrowd', 'masks', 'keypoints', 'image_id']
    for sample_index, (input_tensor, target_dict) in enumerate(train_data_parser):
        if sample_index != 1:
            continue
        (input_tensor, target_dict) = train_data_parser[1357]
        print("Input Shape : ", input_tensor.shape)
        # input_img = torch.from_numpy(np.array([train_data_parser[16][0], train_data_parser[17][0]]))
        input_img = torch.unsqueeze(torch.from_numpy(input_tensor), 0)
        # input_img = torch.ones((2, 3, 448, 448))
        print("Input Shape Unsqueezed : ", input_img.shape)

        # Model Output :
        #       ['boxes', 'labels', 'scores', 'masks']
        result_dict = run_inference(input_img, MODEL_CHKP_PATH)[0]
        print("Result Dict : ", type(result_dict), result_dict.keys())
        print("boxes : ", result_dict['boxes'].shape, type(result_dict['boxes']))
        print("labels : ", result_dict['labels'].shape, type(result_dict['labels']))
        print("scores : ", result_dict['scores'].shape, type(result_dict['scores']))
        print("masks : ", result_dict['masks'].shape, type(result_dict['masks']))

        # Use Non-maximum-suppression to remove extra bounding boxes
        nms_boxes = non_max_suppression_fast(np.array(result_dict['boxes']), 0.5)
        print("NMS Boxes : ", nms_boxes.shape)

        # Create binary mask
        masks_thresh = result_dict['masks']
        masks_thresh[masks_thresh >= 0.5] = 1
        masks_thresh[masks_thresh < 0.5] = 0
        print(np.unique(masks_thresh))

        drawing_input = pi_drawing.draw_input(
            input_tensor=input_tensor,
            **drawing_kwargs
        )

        plt.figure("Input Image")
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

            plt.figure("Semantic Map")
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

            plt.figure("Mask-RCNN Output")
            plt.imshow(drawing_instances)

        plt.show()
        break
