"""
PyTorch script for model training (Variational Autoencoder).

Copyright (C) 2020 by Akira TAMAMORI

This program is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import os
import glob
import sys

import numpy
import torch
import torch.utils.data
from torch import optim, nn
from torch.utils.data.dataset import Subset
from torchsummary import summary

from pytorch_model import VariationalAutoEncoder as VAE
import common as com

# global variable
PARAM = com.yaml_load('./vae.yaml')

# string constant: "cuda:0" or "cpu"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DcaseDataset(torch.utils.data.Dataset):
    """
    Prepare dataset for task2
    """

    def __init__(self, data_dir, dir_name="train", ext="wav"):

        com.logger.info("target_dir : %s", data_dir)

        file_list_path = os.path.abspath(
            "{data_dir}/{dir_name}/*.{ext}".format(data_dir=data_dir,
                                                   dir_name=dir_name,
                                                   ext=ext))
        files = sorted(glob.glob(file_list_path))
        if not files:
            com.logger.exception("no_wav_file!!")

        com.logger.info("train_file num : %s", len(files))

        for file_id, file_name in enumerate(files):
            vector_array = com.file_to_vector_array(
                file_name,
                n_mels=PARAM["feature"]["n_mels"],
                frames=PARAM["feature"]["frames"],
                n_fft=PARAM["feature"]["n_fft"],
                hop_length=PARAM["feature"]["hop_length"],
                power=PARAM["feature"]["power"])

            if file_id == 0:
                dataset_array = numpy.zeros(
                    (vector_array.shape[0] * len(files),
                     PARAM["feature"]["n_mels"] * PARAM["feature"]["frames"]),
                    numpy.float32)

            dataset_array[vector_array.shape[0] * file_id:
                          vector_array.shape[0] * (file_id + 1), :] = vector_array

        self.feat_data = dataset_array

    def __len__(self):
        return self.feat_data.shape[0]  # return num of samples

    def __getitem__(self, index):
        sample = self.feat_data[index, :]  # return vector
        return sample


def training(model, data_loader, optimizer, criterion):
    """
    perform trainging
    """
    model.train()  # training mode
    train_loss = 0.0
    for data in data_loader:
        data = data.to(DEVICE)  # send data to GPU
        data = data.float()  # workaround
        optimizer.zero_grad()  # reset gradient
        loss, xent_loss, kl_loss = model.get_loss(criterion, data)
        # print('total=%f, xent_loss=%f, kl_loss=%f' % (loss, xent_loss, kl_loss))
        loss.backward()  # backpropagation
        train_loss += loss.item()
        optimizer.step()  # update paramerters

    return train_loss / len(data_loader)


def validation(model, data_loader, criterion):
    """
    perform validation
    """
    model.eval()  # validation mode
    val_loss = 0.0
    with torch.no_grad():
        for data in data_loader:
            data = data.to(DEVICE)  # send data to GPU
            data = data.float()  # workaround
            loss, _, _ = model.get_loss(criterion, data)
            val_loss += loss.item()

    return val_loss / len(data_loader)


def main():
    """
    perform model training and validation
    """

    # check mode
    # "development": mode == True
    # "evaluation": mode == False
    mode = com.command_line_chk()  # constant: True or False
    if mode is None:
        sys.exit(-1)

    # make output directory
    os.makedirs(PARAM["model_directory"], exist_ok=True)

    # load base_directory list
    dir_list = com.select_dirs(param=PARAM, mode=mode)

    # loop of the base directory (for each machine)
    dir_list = ['/work/tamamori/dcase2020/dcase2020_task2_baseline/dev_data/ToyCar']
    for idx, target_dir in enumerate(dir_list):
        com.logger.info("===========================")
        com.logger.info("[%d/%d] %s", idx + 1, len(dir_list), target_dir)

        com.logger.info("============== DATASET_GENERATOR ==============")
        dcase_dataset = DcaseDataset(target_dir)
        n_samples = len(dcase_dataset)  # total number of frames
        train_size = int(n_samples * (1.0 - PARAM["training"]["validation_split"]))
        dataset = {"train": None, "val": None}
        dataset["train"] = Subset(dcase_dataset, list(range(0, train_size)))
        dataset["val"] = Subset(dcase_dataset, list(range(train_size, n_samples)))

        com.logger.info("============== DATALOADER_GENERATOR ==============")
        data_loader = {"train": None, "val": None}
        data_loader["train"] = torch.utils.data.DataLoader(
            dataset["train"], batch_size=PARAM["training"]["batch_size"],
            shuffle=PARAM["training"]["shuffle"], drop_last=True)

        data_loader["val"] = torch.utils.data.DataLoader(
            dataset["val"], batch_size=PARAM["training"]["batch_size"],
            shuffle=False, drop_last=False)

        com.logger.info("============== MODEL TRAINING ==============")
        model = VAE(x_dim=PARAM["feature"]["n_mels"] * PARAM["feature"]["frames"],
                    h_dim=PARAM["model"]["hidden_dim"],
                    z_dim=PARAM["model"]["latent_dim"],
                    n_hidden=PARAM["model"]["n_hidden"]).to(DEVICE)

        optimizer = optim.Adam(model.parameters(),
                               weight_decay=PARAM["training"]["weight_decay"])
        criterion = nn.MSELoss(reduction='mean')

        summary(model, input_size=(PARAM["feature"]["n_mels"] *
                                   PARAM["feature"]["frames"],))

        loss = {"train": 0.0, "val": 0.0}
        for epoch in range(1, PARAM["training"]["epochs"] + 1):
            loss["train"] = training(
                model, data_loader["train"], optimizer, criterion)

            loss["val"] = validation(
                model, data_loader["train"], criterion)

            com.logger.info("Epoch %2d: Average train_loss: %.6f, "
                            "Average validation_loss: %.6f",
                            epoch, loss["train"], loss["val"])

        com.logger.info("============== SAVE MODEL ==============")
        torch.save(model.state_dict(),
                   "%s/model_ae_%s.pt" % (PARAM["model_directory"],
                                          os.path.split(target_dir)[1]))
        com.logger.info("save_model -> %s",
                        "%s/model_ae_%s.pt" % (PARAM["model_directory"],
                                               os.path.split(target_dir)[1]))
        com.logger.info("============== END TRAINING ==============")


if __name__ == "__main__":
    main()
