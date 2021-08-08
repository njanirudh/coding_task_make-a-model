import pytorch_lightning as pl
import torch
import torchvision
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from src.pidata.pidata import pi_parser
from src.utils.custom_config import custom_parser_config
from utils.checkpoint_utils import PeriodicCheckpoint

# Setting seed for reproducibility
seed = 666
torch.manual_seed(seed)

Tensor = torch.tensor
Module = torch.nn.Module


def get_model_instance_segmentation(num_classes: int):
    """
    Returns MaskRCNN model
    Args:
        num_classes: Total number of classes.

    Returns: Mask-RCNN Model
    """
    # load an instance segmentation model pre-trained pre-trained on COCO and Imagenet.
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True,
                                                               pretrained_backbone=True,
                                                               trainable_backbone_layers=0,
                                                               )

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model


class SegmentationModule(pl.LightningModule):
    """
    Pytorch Lightning module for training
    different segmentation models.
    """

    def __init__(self, config_data: dict, train_mode: bool = True,
                 batch_size: int = 5, gpu: int = 1,
                 epochs: int = 50, lr: float = 0.003) -> None:
        """
        :param config_data: Config for data, utils.
        :param train_mode: Sets model to training or run_inference mode.
        :param batch_size: Batch size during training.
        :param epochs: Total epochs for training. (default 50)
        :param lr: Learning rate (default 0.003)
        :param gpu: Set total gpus to use (default 1)
        """
        super(SegmentationModule, self).__init__()

        # (1)-> Testing custom model
        # self.model = UNET(3, 4)

        # (2)-> Default Mask-RCNN model
        self.model = get_model_instance_segmentation(4)

        # (3)-> Model from 'segmentation_models_pytorch' library
        # self.model = smp.Unet(
        #     encoder_name="mobilenet_v2",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        #     encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        #     in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        #     classes=4,  # model output channels (number of classes in your dataset)
        # )

        self.model.train(train_mode)

        self.val_loader, self.train_loader = None, None
        self.num_train_imgs, self.num_val_imgs = None, None
        self.trainer, self.curr_device = None, None

        # Model checkpoint saving every 500 steps
        self.periodic_chkp = PeriodicCheckpoint(1000)
        self.save_hyperparameters()

        self.config_data = config_data
        self.loss_fn = CrossEntropyLoss()
        self.batch_size = batch_size
        self.epochs = epochs
        self.samples = 1000
        self.learning_rate = lr
        self.gpu = gpu

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input)

    def training_step(self, batch, batch_idx):
        print("Batch ==>", batch_idx, len(batch))

        inputs, labels = batch

        loss_dict = self.model(inputs, labels)
        print("Training Loss :: ", len(loss_dict), loss_dict.keys())
        losses = sum(loss for loss in loss_dict.values())

        return losses

    def validation_step(self, batch, batch_idx):
        print("Batch ==>", batch_idx, len(batch))

        inputs, labels = batch

        loss_dict = self.model(inputs, labels)
        print("Loss :: ", len(loss_dict), loss_dict.keys())

        return loss_dict

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def train_dataloader(self):
        training_data_parser = pi_parser.PiParser(
            config=custom_parser_config,
            split_name="train",
            num_samples=self.samples,  # number of samples to be drawn, set to a multiple of the batch size
            numpy_to_tensor_func=torch.from_numpy,
            # framework-dependent, e.g. torch.from_numpy (PyTorch), if None, the returned type is numpy.ndarray
        )

        self.train_loader = DataLoader(training_data_parser,
                                       batch_size=self.batch_size,
                                       shuffle=True,
                                       num_workers=4,
                                       collate_fn=self.collate_fn  # custom collate
                                       )
        self.num_train_imgs = len(self.train_loader)
        return self.train_loader

    def val_dataloader(self):
        val_data_parser = pi_parser.PiParser(
            config=custom_parser_config,
            split_name="train",
            num_samples=self.samples,  # number of samples to be drawn, set to a multiple of the batch size
            numpy_to_tensor_func=torch.from_numpy,
            # framework-dependent, e.g. torch.from_numpy (PyTorch), if None, the returned type is numpy.ndarray
        )

        self.val_loader = DataLoader(val_data_parser,
                                     batch_size=self.batch_size,
                                     shuffle=True,
                                     num_workers=4,
                                     collate_fn=self.collate_fn  # custom collate
                                     )
        self.num_val_imgs = len(self.val_loader)
        return self.val_loader

    def train_model(self):
        self.trainer = pl.Trainer(gpus=self.gpu,
                                  max_epochs=self.epochs,
                                  callbacks=self.periodic_chkp,
                                  weights_summary="full",
                                  # overfit_batches=2,
                                  accumulate_grad_batches=5,
                                  # check_val_every_n_epoch=10,
                                  stochastic_weight_avg=True
                                  )
        self.trainer.fit(self,
                         self.train_dataloader(),
                         self.val_dataloader())

    def collate_fn(self, batch):
        image_list, target_list = [], []
        for item in batch:
            images, targets = item

            image_list.append(images)
            target_list.append(targets)

        # print("Img ->", len(image_list), type(image_list))
        # print("Targets ->", len(target_list), type(target_list))
        return image_list, target_list


if __name__ == "__main__":
    model_trainer = SegmentationModule(config_data=custom_parser_config,
                                       train_mode=True,
                                       lr=0.0001,
                                       batch_size=2,
                                       epochs=500,
                                       gpu=1)
    model_trainer.train_model()
