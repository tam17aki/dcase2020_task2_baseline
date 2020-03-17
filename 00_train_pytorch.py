"""
 @file   00_train.py
 @brief  Script for training
 @author Toshiki Nakamura, Yuki Nikaido, and Yohei Kawaguchi (Hitachi Ltd.)
 Copyright (C) 2020 Hitachi, Ltd. All right reserved.
"""

########################################################################
# import default python-library
########################################################################
import os
import glob
import sys
########################################################################


########################################################################
# import additional python-library
########################################################################
import numpy
import common as com
import torch
import torch.utils.data
from torch import optim, nn
from torch.utils.data.dataset import Subset
from pytorch_model import AutoEncoder

########################################################################
# load parameter.yaml
########################################################################
param = com.yaml_load('./baseline.yaml')
########################################################################


########################################################################
# Dataset
########################################################################
class dcaseDataset(torch.utils.data.Dataset):
    def __init__(self, target_dir, dir_name="train", ext="wav",
                 n_mels=64, frames=5, n_fft=1024, hop_length=512, power=2.0,
                 transform=None):
        self.transform = transform

        com.logger.info("target_dir : {}".format(target_dir))

        file_list_path = os.path.abspath(
            "{dir}/{dir_name}/*.{ext}".format(dir=target_dir, dir_name=dir_name,
                                              ext=ext))
        files = sorted(glob.glob(file_list_path))
        if len(files) == 0:
            com.logger.exception("no_wav_file!!")

        com.logger.info("train_file num : {num}".format(num=len(files)))

        dims = n_mels * frames
        for idx in range(len(files)):
            vector_array = com.file_to_vector_array(files[idx],
                                                    n_mels=n_mels,
                                                    frames=frames,
                                                    n_fft=n_fft,
                                                    hop_length=hop_length,
                                                    power=power)
            if idx == 0:
                dataset = numpy.zeros(
                    (vector_array.shape[0] * len(files), dims), float)

            dataset[vector_array.shape[0] * idx:
                    vector_array.shape[0] * (idx + 1), :] = vector_array

        self.feat_data = dataset

    def __len__(self):
        return self.feat_data.shape[0]  # return num of samples

    def __getitem__(self, index):
        sample = self.feat_data[index, :]  # return vector

        if self.transform:
            sample = self.transform(sample)

        return sample


########################################################################
# main 00_train.py
########################################################################
if __name__ == "__main__":
    # check mode
    # "development": mode == True
    # "evaluation": mode == False
    mode = com.command_line_chk()
    if mode is None:
        sys.exit(-1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # make output directory
    os.makedirs(param["model_directory"], exist_ok=True)

    # load base_directory list
    dirs = com.select_dirs(param=param, mode=mode)

    # loop of the base directory (for each machine)
    for idx, target_dir in enumerate(dirs):
        com.logger.info("===========================")
        com.logger.info("[{idx}/{total}] {dirname}".format(
            dirname=target_dir, idx=idx+1, total=len(dirs)))

        # set path
        machine_type = os.path.split(target_dir)[1]

        model_file_path = "{model}/model_ae_{machine_type}.pt".format(
            model=param["model_directory"],
            machine_type=machine_type)

        com.logger.info("============== DATASET_GENERATOR ==============")
        dataset = dcaseDataset(target_dir,
                               n_mels=param["feature"]["n_mels"],
                               frames=param["feature"]["frames"],
                               n_fft=param["feature"]["n_fft"],
                               hop_length=param["feature"]["hop_length"],
                               power=param["feature"]["power"],)
        n_samples = len(dataset)
        train_size = int(n_samples * (1.0 - param["fit"]["validation_split"]))
        subset1_indices = list(range(0, train_size))
        subset2_indices = list(range(train_size, n_samples))
        train_dataset = Subset(dataset, subset1_indices)
        val_dataset = Subset(dataset, subset2_indices)

        com.logger.info("============== DATALOADER_GENERATOR ==============")
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=param["fit"]["batch_size"],
            shuffle=param["fit"]["shuffle"],
            drop_last=True)

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=param["fit"]["batch_size"],
            shuffle=False,
            drop_last=False)

        com.logger.info("============== MODEL TRAINING ==============")
        model = AutoEncoder().to(device)

        optimizer = optim.Adam(model.parameters())
        criterion = nn.MSELoss()

        epochs = param["fit"]["epochs"]
        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0
            for batch_idx, data in enumerate(train_loader):
                data = data.to(device)  # send data to GPU
                data = data.float()  # workaround
                optimizer.zero_grad()
                reconst = model(data)  # reconstruction through auto encoder
                loss = criterion(data, reconst)  # mean squared error
                loss.backward()  # backpropagation
                train_loss += loss.item()
                optimizer.step()  # update paramerters

            model.eval()  # freeze temporarily
            val_loss = 0
            with torch.no_grad():
                for batch_idx, data in enumerate(val_loader):
                    data = data.to(device)  # send data to GPU
                    data = data.float()  # workaround
                    reconst = model(data)  # reconstruction through auto encoder
                    loss = criterion(data, reconst)  # mean squared error
                    val_loss += loss.item()

            # average loss over whole mini-batches
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            com.logger.info('Epoch: {} Average train_loss: {:.6f}, '
                            'Average val_loss: {:.6f}'.format(epoch, train_loss,
                                                              val_loss))

        # save models
        torch.save(model.state_dict(), model_file_path)
        com.logger.info("save_model -> {}".format(model_file_path))
        com.logger.info("============== END TRAINING ==============")
