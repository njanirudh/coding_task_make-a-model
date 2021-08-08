import cv2
import torch
import numpy as np
import random

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

    MODEL_CHKP_PATH = "/home/anirudh/NJ/Interview/Pheno-Inspect/git_proj/" \
                          "coding_task_make-a-model/src/lightning_logs/version_1/" \
                          "checkpoints/epoch=92-step=184.ckpt"

    image = torch.ones((1,3,448,448))
    print(image.shape)

    result = inference(image, MODEL_CHKP_PATH)
    print(len(result))

    result = result[0]
    print(result.keys())
    print(result['boxes'].shape)
    print(result['labels'].shape)
    print(result['scores'].shape)
    print(result['masks'].shape)