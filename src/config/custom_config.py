custom_parser_config = {
    "datasets": {
        "train": [
            {
                "path": "../../dataset/sugarbeet_weed_dataset",  # TODO adjust as required
                "sampling_weight": 1.0,
            }
        ],
        "val": [{"path": "../../dataset/sugarbeet_weed_dataset",
                 "sampling_weight": 1.0}],
        "test": [{"path": "../../dataset/sugarbeet_weed_dataset",
                  "sampling_weight": 1.0}],
    },
    "input_layers": [
        {
            "name": "rgb",
            "channels": 3,
            "mean": [0.485, 0.456, 0.406],  # input data normalization
            "std": [0.229, 0.224, 0.225],  # using imagenet values
        }
    ],
    "instance_filter": {"min_box_area": 100, "min_mask_area": 100},
    "model_input": {"height": 224, "width": 224},  # TODO adjust as required
    "model_output": {
        "height": 224,  # TODO adjust as required
        "offset_x": 0,
        "offset_y": 0,
        "stride_x": 1,
        "stride_y": 1,
        "width": 224,
    },
    "required_targets": {
        "area": True,
        "boxes": True,
        "iscrowd": True,
        "keypoints": True,
        "labels": True,
        "masks": True,
        "semantics": True,
    },
    "samplers": {
        "train": [
            {
                "uniform": {
                    "offset_from_boundary_x": 0.0,
                    "offset_from_boundary_y": 0.0,
                    "weight": 0.5,
                }
            },  # samples a patch from anywhere within the image
            {"instances": {"weight": 0.5}},
            # samples a patch from image regions with plant instances
            # the frequency of each class is determined by the 'sampling_weight',
            # see semantic labels
        ],
        "val": [
            {
                "uniform": {
                    "offset_from_boundary_x": 0.0,
                    "offset_from_boundary_y": 0.0,
                    "weight": 0.5,
                }
            },
            {"instances": {"weight": 0.5}},
        ],
        "test": [
            {
                "uniform": {
                    "offset_from_boundary_x": 0.0,
                    "offset_from_boundary_y": 0.0,
                    "weight": 0.5,
                }
            },
            {"instances": {"weight": 0.5}},
        ],
    },
    "seed": {"test": 2, "train": 0, "val": 1},
    "semantic_labels": [
        {
            "color": [0, 0, 0],  # color for visualization
            "has_instances": False,
            "join_with": [
                "plant.Unlabeled",
                "plant.Vegetation",
            ],  # unrecognized vegetation is mapped to Soil/Background
            "name": "plant.Soil",
            "sampling_weight": 1.0,
        },
        {
            "color": [0, 255, 0],
            "has_instances": True,
            "join_with": [],
            "name": "plant.Sugarbeet",
            "sampling_weight": 1.0,
        },
        {
            "color": [255, 0, 0],
            "has_instances": True,
            "join_with": [],
            "name": "plant.Weed",
            "sampling_weight": 1.0,
        },
    ],
    "transforms": {
        "train": [
            {
                "affine": {
                    "flip_x_probability": 0.5,
                    "flip_y_probability": 0.5,
                    "probability": 1.0,
                    "rotation_max": 3.141592653589793,
                    "rotation_min": -3.141592653589793,
                    "scaling_x_max": 1.25,
                    "scaling_x_min": 0.8,
                    "scaling_y_max": 1.25,
                    "scaling_y_min": 0.8,
                    "shearing_x_max": 0.2,
                    "shearing_x_min": -0.2,
                    "shearing_y_max": 0.2,
                    "shearing_y_min": -0.2,
                    "translation_x_max": 224.0,
                    "translation_x_min": -224.0,
                    "translation_y_max": 224.0,
                    "translation_y_min": -224.0,
                }
            },
            {
                "hsv": {
                    "channels": [0, 1, 2],
                    "hue_max": 0.05,
                    "hue_min": -0.05,
                    "probability": 1.0,
                    "saturation_max": 0.05,
                    "saturation_min": -0.05,
                    "value_max": 0.1,
                    "value_min": -0.1,
                }
            },
            {
                "contrast": {
                    "channels": [0, 1, 2],
                    "contrast_max": 0.1,
                    "contrast_min": -0.1,
                    "probability": 1.0,
                }
            },
            {
                "blur": {
                    "blur_max": 2.0,
                    "blur_min": 0.0,
                    "channels": [0, 1, 2],
                    "probability": 0.2,
                }
            },
        ],
        "val": {},  # no random transorms for valdation and test case
        "test": {},
    },
}
