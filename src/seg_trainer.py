from pprint import pprint

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn.functional import log_softmax
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

logger = TensorBoardLogger("tb_logs", name="my_model")

from utils.checkpoint_utils import PeriodicCheckpoint

from src.model.unet import UNET
from src.pidata.pidata import pi_parser
from src.utils.custom_config import custom_parser_config

# Setting seed for reproducibility
seed = 666
torch.manual_seed(seed)

Tensor = torch.tensor
Module = torch.nn.Module


class SegmentationModule(pl.LightningModule):
    """
    Pytorch Lightning module for training the
    UNet segmentation model.
    """

    def __init__(self, config_data: dict, train_mode: bool = True,
                 batch_size: int = 10, gpu: int = 1,
                 epochs: int = 50, lr: float = 0.003) -> None:
        """
        :param config_data: Config for data, utils.
        :param train_mode: Sets model to training or inference mode.
        :param batch_size: Batch size during training.
        :param epochs: Total epochs for training. (default 50)
        :param lr: Learning rate (default 0.003)
        :param gpu: Set total gpus to use (default 1)
        """
        super(SegmentationModule, self).__init__()

        # Testing custom model
        # self.model = UNET(3, 3)
        # self.model.train(train_mode)  # Set training mode = true

        # Model from 'segmentation_models_pytorch' library
        self.model = smp.Unet(
            encoder_name="mobilenet_v2",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=4,  # model output channels (number of classes in your dataset)
        )
        self.model.train(train_mode)

        self.val_loader, self.train_loader = None, None
        self.num_train_imgs, self.num_val_imgs = None, None
        self.trainer, self.curr_device = None, None

        # Model checkpoint saving every 500 steps
        self.periodic_chkp = PeriodicCheckpoint(500)

        self.config_data = config_data
        self.loss_fn = CrossEntropyLoss()
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = lr
        self.gpu = gpu

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input)

    def training_step(self, batch, batch_idx):
        # print("Batch ==>", len(batch))
        inputs, labels = batch
        # inputs, labels = inputs, labels['semantics']
        self.curr_device = inputs.device

        # print(inputs.shape, labels.shape)

        outputs = self.forward(inputs)
        # print("Output -->", outputs.shape)
        # outputs = torch.argmax(outputs, 1)
        # print("Outputs -->", outputs.shape)
        # print(outputs.shape, labels.shape)

        train_loss = self.loss_fn(log_softmax(outputs.float(), 0), labels)

        return train_loss

    def validation_step(self, batch, batch_idx):
        # print("Batch ==>", len(batch[1]))

        inputs, labels = batch
        # inputs, labels = inputs, labels['semantics']
        self.curr_device = inputs.device

        # print(inputs.shape, labels.shape)

        outputs = self.forward(inputs)
        # print("Output -->", outputs.shape)
        # outputs = torch.argmax(outputs, 1)
        # print("Output -->", outputs.shape)
        # print(outputs.shape, labels.shape)
        # print(np.unique(outputs), np.unique(labels))
        val_loss = self.loss_fn(outputs, labels)

        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def train_dataloader(self):
        training_data_parser = pi_parser.PiParser(
            config=custom_parser_config,
            split_name="train",
            num_samples=10,  # number of samples to be drawn, set to a multiple of the batch size
            numpy_to_tensor_func=None,
            # framework-dependent, e.g. torch.from_numpy (PyTorch), if None, the returned type is numpy.ndarray
        )

        self.train_loader = DataLoader(training_data_parser,
                                       batch_size=self.batch_size,
                                       shuffle=True,
                                       num_workers=4,
                                       collate_fn=self.collate_fn
                                      )
        self.num_train_imgs = len(self.train_loader)
        return self.train_loader

    def val_dataloader(self):
        val_data_parser = pi_parser.PiParser(
            config=custom_parser_config,
            split_name="train",
            num_samples=10,  # number of samples to be drawn, set to a multiple of the batch size
            numpy_to_tensor_func=None,
            # framework-dependent, e.g. torch.from_numpy (PyTorch), if None, the returned type is numpy.ndarray
        )

        self.val_loader = DataLoader(val_data_parser,
                                     batch_size=self.batch_size,
                                     shuffle=True,
                                     num_workers=4,
                                     collate_fn=self.collate_fn
                                     )
        self.num_val_imgs = len(self.val_loader)
        return self.val_loader

    def train_model(self):
        self.trainer = pl.Trainer(gpus=self.gpu, max_epochs=self.epochs,
                                  callbacks=self.periodic_chkp)
        self.trainer.fit(self,
                         self.train_dataloader(),
                         self.val_dataloader())

    def collate_fn(self, batch):

        input_data_list = []
        label_list = []
        for item in batch:
            input_tensor, target_dict = item
            # print(type(input_tensor), type(target_dict))
            # print(len(input_tensor), len(target_dict))
            inputs, labels = input_tensor, target_dict['semantics']
            input_data_list.append(inputs)
            label_list.append(labels)
        # zipped = zip(input_data_list, label_list)
        return torch.from_numpy(np.array(input_data_list)), \
               torch.from_numpy(np.array(label_list))


if __name__ == "__main__":
    model_trainer = SegmentationModule(config_data=custom_parser_config,
                                       batch_size=5,
                                       epochs=150,
                                       gpu=1)
    model_trainer.train_model()
