"""
PyTorch script for model training (Generative Adversarial Nets).

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

from pytorch_model import Generator, Discriminator
import common as com

# global variable
PARAM = com.yaml_load('./gan.yaml')

# string constant: "cuda:0" or "cpu"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# label for GAN
REAL_LABEL = 1
FAKE_LABEL = 0


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
    Perform trainging
    """
    model_g = model["Generator"]
    model_d = model["Discriminator"]
    model_g.train()  # training mode
    model_d.train()  # training mode

    criterion_g = criterion["Generator"]
    criterion_d = criterion["Discriminator"]

    gan_loss = {"generator": 0.0, "discriminator": 0.0}

    for i, data in enumerate(data_loader, 0):
        data = data.to(DEVICE)  # send data to GPU
        data = data.float()  # workaround

        mini_batch_size = data.size()[0]
        label_real = torch.full((mini_batch_size, 1), REAL_LABEL, device=DEVICE)
        label_fake = torch.full((mini_batch_size, 1), FAKE_LABEL, device=DEVICE)

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # Train with all-real batch
        optimizer["Discriminator"].zero_grad()

        # Forward pass real batch through D
        output = model_d(data)

        # Calculate loss on all-real batch
        err_d_real = criterion_d(output, label_real)

        # Calculate gradients for D in backward pass
        err_d_real.backward()
        discrimitor_outx = output.mean().item()  # D(x)

        # Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn((mini_batch_size, PARAM["model"]["latent_dim"]),
                            device=DEVICE)
        fake = model_g(noise)

        # Classify all fake batch with D
        output = model_d(fake.detach())
        # detach() is essential to prevent the gradient from flowing into G.

        # Calculate D's loss on the all-fake batch
        err_d_fake = criterion_d(output, label_fake)
        # Calculate the gradients for this batch
        err_d_fake.backward()

        discrimitor_gz = output.mean().item()  # D(G(z))

        err_d = err_d_real + err_d_fake
        gan_loss["discriminator"] += err_d.item()

        # Update discriminator
        optimizer["Discriminator"].step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        optimizer["Generator"].zero_grad()

        # Since we just updated D, perform another forward pass
        # of all-fake batch through D
        output = model_d(fake)

        # Calculate G's loss based on this output
        err_g = criterion_g(output, label_real)

        # Calculate gradients for G
        err_g.backward()
        gan_loss["generator"] += err_g.item()

        discrimitor_gz2 = output.mean().item()  # D(G(z))

        # Update generator
        optimizer["Generator"].step()

        if i % 50 == 0:
            com.logger.info(
                '[%d/%d]\tLoss_D: %.6f\tLoss_G: %.6f'
                '\tD(x): %.6f\tD(G(z)): %.6f / %.6f',
                i, len(data_loader), err_d.item(), err_g.item(),
                discrimitor_outx, discrimitor_gz, discrimitor_gz2)

    gan_loss["discriminator"] /= len(data_loader)
    gan_loss["generator"] /= len(data_loader)

    return gan_loss["discriminator"], gan_loss["generator"]


def validation(model, data_loader, criterion):
    """
    Perform validation
    """
    model_g = model["Generator"]
    model_d = model["Discriminator"]
    model_g.eval()  # validation mode
    model_d.eval()  # validation mode
    criterion_g = criterion["Generator"]
    criterion_d = criterion["Discriminator"]
    val_loader = data_loader["val"]
    val_loss_d = 0.0
    val_loss_g = 0.0
    with torch.no_grad():
        for data in val_loader:
            mini_batch_size = data.size()[0]
            label_real = torch.full((mini_batch_size, 1), REAL_LABEL, device=DEVICE)
            label_fake = torch.full((mini_batch_size, 1), FAKE_LABEL, device=DEVICE)

            data = data.to(DEVICE)  # send data to GPU
            data = data.float()  # workaround
            output = model_d(data)
            err_d_real = criterion_d(output, label_real)

            noise = torch.randn((mini_batch_size, PARAM["model"]["latent_dim"]),
                                device=DEVICE)
            fake = model_g(noise)
            output = model_d(fake)
            err_d_fake = criterion_d(output, label_fake)

            val_loss_d += err_d_real.item() + err_d_fake.item()

            output = model_d(fake)
            err_g = criterion_g(output, label_real)
            val_loss_g += err_g.item()

    return val_loss_d / len(val_loader), val_loss_g / len(val_loader)


def main():
    """
    Perform model training and validation
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
        model = {}
        model["Generator"] = Generator(
            x_dim=PARAM["feature"]["n_mels"] * PARAM["feature"]["frames"],
            h_dim=PARAM["model"]["hidden_dim"],
            z_dim=PARAM["model"]["latent_dim"]).to(DEVICE)

        model["Discriminator"] = Discriminator(
            x_dim=PARAM["feature"]["n_mels"] * PARAM["feature"]["frames"],
            h_dim=PARAM["model"]["hidden_dim"],
            z_dim=PARAM["model"]["latent_dim"],).to(DEVICE)

        optimizer = {}
        optimizer["Generator"] = optim.Adam(
            model["Generator"].parameters(),
            lr=PARAM["training"]["learning_rate"],
            eps=0.0001,
            weight_decay=PARAM["training"]["weight_decay"])
        optimizer["Discriminator"] = optim.Adam(
            model["Discriminator"].parameters(),
            lr=PARAM["training"]["learning_rate"],
            eps=0.0001,
            weight_decay=PARAM["training"]["weight_decay"])

        scheduler = {}
        scheduler["Generator"] = optim.lr_scheduler.StepLR(
            optimizer["Generator"],
            step_size=PARAM["training"]["lr_step_size"],
            gamma=PARAM["training"]["lr_gamma"])
        scheduler["Discriminator"] = optim.lr_scheduler.StepLR(
            optimizer["Discriminator"],
            step_size=PARAM["training"]["lr_step_size"],
            gamma=PARAM["training"]["lr_gamma"])

        criterion = {}
        criterion["Generator"] = nn.BCELoss()
        criterion["Discriminator"] = nn.BCELoss()

        loss = {"train_G": 0.0, "train_D": 0.0, "val_G": 0.0, "val_D": 0.0}

        for epoch in range(1, PARAM["training"]["epochs"] + 1):
            loss["train_D"], loss["train_G"] = training(
                model, data_loader["train"], optimizer, criterion)

            scheduler["Generator"].step()
            scheduler["Discriminator"].step()

            loss["val_D"], loss["val_G"] = validation(
                model, data_loader, criterion)

            com.logger.info("Epoch %2d: train_loss(D): %.6f, "
                            "train_loss(G): %.6f, "
                            "val_loss(D): %.6f, "
                            "train_loss(G): %.6f",
                            epoch, loss["train_D"], loss["train_G"],
                            loss["val_D"], loss["val_G"])

        com.logger.info("============== SAVE MODEL ==============")
        torch.save(model["Generator"].state_dict(),
                   "%s/model_generator_%s.pt" % (PARAM["model_directory"],
                                                 os.path.split(target_dir)[1]))
        torch.save(model["Discriminator"].state_dict(),
                   "%s/model_discriminator_%s.pt" % (PARAM["model_directory"],
                                                     os.path.split(target_dir)[1]))
        com.logger.info("save_model -> %s", "%s/model_generator_%s.pt"
                        % (PARAM["model_directory"], os.path.split(target_dir)[1]))
        com.logger.info("save_model -> %s", "%s/model_discriminator_%s.pt"
                        % (PARAM["model_directory"], os.path.split(target_dir)[1]))
        com.logger.info("============== END TRAINING ==============")


if __name__ == "__main__":
    main()
