import matplotlib
import torch

from src.pidata.pidata import pi_parser
from src.utils.custom_config import custom_parser_config

matplotlib.use('TKAgg', warn=False, force=True)


def collate_fn(self, batch):
    return tuple(zip(*batch))


if __name__ == "__main__":
    dataset = pi_parser.PiParser(
        config=custom_parser_config,
        split_name="train",
        num_samples=2,  # number of samples to be drawn, set to a multiple of the batch size
        numpy_to_tensor_func=None,
        # framework-dependent, e.g. torch.from_numpy (PyTorch), if None, the returned type is numpy.ndarray
    )

    # torch.manual_seed(1)
    # indices = torch.randperm(len(dataset)).tolist()
    # dataset = torch.utils.data.Subset(dataset, indices[:-50])
    # dataset_test = torch.utils.data.Subset(dataset, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=4,
        collate_fn=collate_fn)

    print(len(dataset))
    print(len(data_loader))
    #
    # for i,j in enumerate(dataset):
    #     print(i, np.unique(j[1]['semantics']))

    # # print(type(dataset[0]), len(dataset))
    # #
    # semantics = dataset[16][1]['semantics']
    # # semantics = np.where(semantics == 0, 10, semantics)
    # print(semantics.shape)
    # print(np.unique(semantics))
    # plt.imshow(semantics, cmap='Blues')
    # plt.show()
