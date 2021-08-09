import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from src.utils.custom_config import custom_parser_config

from seg_trainer import SegmentationModule

matplotlib.use('TKAgg', warn=False, force=True)

if __name__ == "__main__":
    input_img = cv2.imread(
        "/home/anirudh/NJ/Interview/Pheno-Inspect/git_proj/coding_task_make-a-model/dataset/sugarbeet_weed_dataset/items/"
        "68653b6d-f406-442d-833e-31ffb43cf578/map/tileLayers/rgb/tiles/0-0-1.png")
    input_img = input_img[np.newaxis, ...]
    input_img = np.swapaxes(input_img, 1, 3)
    input_img = torch.from_numpy(input_img).float()

    MODEL_CHKP_PATH = "/home/anirudh/NJ/Interview/Pheno-Inspect/git_proj/coding_task_make-a-model/src/" \
                      "lightning_logs/version_198/checkpoints/epoch=91-step=183.ckpt"
    seg_inference = SegmentationModule(config_data=custom_parser_config,
                                       batch_size=5,
                                       epochs=150,
                                       gpu=1,
                                       train_mode=False)
    seg_inference.model.eval()

    seg_inference.load_state_dict(torch.load(MODEL_CHKP_PATH), strict=False)
    with torch.no_grad():
        print(input_img.shape)
        output_seg = seg_inference(input_img)
        output_seg = torch.argmax(output_seg, 1)

        print(output_seg.shape)
        print(np.unique(output_seg))

        output_seg = np.swapaxes(output_seg, 0, 2)
        print(np.unique(output_seg))
        result = np.squeeze(output_seg)
        plt.imshow(result, cmap='Blues')
        plt.show()
