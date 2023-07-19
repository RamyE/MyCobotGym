import os
import sys
import numpy as np
import torch
import wandb
import argparse

from torch import nn, optim
from torchvision import transforms
from model import ObjectLocalization
from mycobotgym.obj_localize.utils.general_utils import *
from mycobotgym.obj_localize.utils.train_utils import *
from mycobotgym.obj_localize.constants import *


def main(epochs, file_name, pre_train=0, use_wandb=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    if use_wandb:
        run = wandb_init()

    if pre_train != 0:
        weights_path = "{}/{}/VGGNet_epoch{}.pth".format(MODEL_DIR, file_name, pre_train - 1)
        print(weights_path)
        model = ObjectLocalization().to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("Freeze the pretrained weights")
        for param in model.vgg16_features.parameters():
            param.requires_grad = False
    else:
        weights_path = "{}/VGG16_weights.pth".format(MODEL_DIR)
        vgg16_weights = load_weights(weights_path)
        model = ObjectLocalization(vgg16_weights, freeze=True).to(device)

    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    batch_size = 25
    file_dir = os.path.join(ROOT_DATA_DIR, file_name)
    data_dir = os.path.join(file_dir, "data")
    print(data_dir)
    for _, _, filenames in os.walk(data_dir):
        num_images = len(filenames)
        print(num_images)
    ''''''
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_loader, val_loader = generate_datasets(num_images, file_dir, batch_size, transform)

    train_err_path = "{}/{}/{}_train_err.csv".format(MODEL_DIR, file_name, "VGGNet")
    val_err_path = "{}/{}/{}_val_err.csv".format(MODEL_DIR, file_name, "VGGNet")
    if wandb.run.resumed:
        checkpoint = torch.load(wandb.restore(CHECKPOINT_PATH))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']

    if pre_train != 0:
        train_err = np.loadtxt(train_err_path).tolist()
        val_err = np.loadtxt(val_err_path).tolist()
    else:
        train_err = []
        val_err = []

    for epoch in range(epochs):
        tr_loss, avg_train_loss = train(model, optimizer, loss_function, train_loader, device, epoch, epochs)
        # train_err.append(tr_loss)
        train_err.append(avg_train_loss)

        val_loss, avg_val_loss = validation(model, loss_function, val_loader, device, epoch, epochs)
        # val_err.append(val_loss)
        val_err.append(avg_val_loss)
        # print(train_err)
        # print(val_err)
        np.savetxt(train_err_path, train_err)
        np.savetxt(val_err_path, val_err)
        save_path = f"{MODEL_DIR}/{file_name}/VGGNet_epoch{epoch + pre_train}.pth"
        ckpt_path = f"{MODEL_DIR}/{file_name}/checkpoint.tar"
        wandb.log({"Epoch:": epoch, "train loss": avg_train_loss, "val loss": avg_val_loss})

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, ckpt_path)
        torch.save(model.state_dict(), save_path)
        # wandb.save(CHECKPOINT_PATH)

    wandb.finish()
    return train_err_path, val_err_path


def test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    model = ObjectLocalization().to(device)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # data_dir = os.path.join(ROOT_DATA_DIR, "test_object")
    data_dir = os.path.join(ROOT_DATA_DIR, "test_target")
    test_data = TrainDataset(data_dir, transform)

    # weights_path = 'trained_models/reach_object/VGGNet_epoch29.pth'
    weights_path = 'trained_models/reach_target/VGGNet_epoch49.pth'
    model.load_state_dict(torch.load(weights_path, map_location=device))

    run_test(model, device, test_data, 50)


if __name__ == '__main__':
    # train_pth, val_pth = main(50, "reach_target_xy", use_wandb=True)
    test()

