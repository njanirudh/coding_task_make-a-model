import pytorch_lightning as pl
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from config.custom_config import custom_parser_config
from model.MaskRCNN import get_model_instance_segmentation
from pidata.pidata import pi_parser
from piutils.piutils import pi_log
from utils.checkpoint_utils import PeriodicCheckpoint

logger = pi_log.get_logger(__name__)
logger.propagate = False

# Setting seed for reproducibility
seed = 666
torch.manual_seed(seed)

Tensor = torch.tensor
Module = torch.nn.Module


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

        # Model checkpoint saving every 10 steps
        self.periodic_chkp = PeriodicCheckpoint(10)
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
        print("Training Loss :: ", type(loss_dict), len(loss_dict))
        losses = sum(loss for loss in loss_dict.values())

        return losses

    def validation_step(self, batch, batch_idx):
        print("Batch ==>", batch_idx, len(batch))

        inputs, labels = batch

        loss_dict = self.model(inputs, labels)
        print("Loss :: ", type(loss_dict), len(loss_dict))

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
            split_name="val",
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
                                  # accumulate_grad_batches=2,
                                  check_val_every_n_epoch=10,
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
                                       epochs=10,
                                       gpu=1)
    model_trainer.train_model()
