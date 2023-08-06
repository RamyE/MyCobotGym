import os
import sys
import torch
import numpy as np

from tqdm import tqdm
from torch.utils.data import sampler
from torch.utils.data import DataLoader
from mycobotgym.obj_localize.vision_model.dataset import TrainDataset


def train(model, optimizer, loss_function, tr_loader, device, epoch, epochs, cat_model=False):
    model.train()
    train_bar = tqdm(tr_loader, file=sys.stdout)
    num_itr = 0
    train_acc_loss = 0
    for step, train_data in enumerate(train_bar):
        optimizer.zero_grad()
        if cat_model:
            image1, image2, label = train_data
            output = model(image1.to(device), image2.to(device))
        else:
            image, label = train_data
            output = model(image.to(device))
        train_loss = loss_function(output, label.to(device))
        train_loss.backward()
        optimizer.step()
        train_acc_loss += train_loss.item()
        num_itr += 1
        train_bar.desc = "Epoch[{}/{}] train loss:{}".format(epoch + 1, epochs, train_loss)
    return train_loss.item(), train_acc_loss / num_itr


def validation(model, loss_function, val_loader, device, epoch, epochs, cat_model=False):
    model.eval()
    with torch.no_grad():
        val_bar = tqdm(val_loader, file=sys.stdout)
        num_itr = 0
        val_acc_loss = 0
        for val_data in val_bar:
            if cat_model:
                image1, image2, label = val_data
                output = model(image1.to(device), image2.to(device))
            else:
                image, label = val_data
                output = model(image.to(device))
            val_loss = loss_function(output, label.to(device))
            val_acc_loss += val_loss.item()
            num_itr += 1
            val_bar.desc = "Epoch[{}/{}] val loss:{}".format(epoch + 1, epochs, val_loss)
    return val_loss.item(), val_acc_loss / num_itr


def run_test(model, device, test_data, num_itr=None, log_dir=None, cat_model=False):
    model.eval()
    success = 0
    total_x_err = total_y_err = total_z_err = 0
    if num_itr is None:
        num_itr = len(test_data)
    if log_dir is None:
        log = sys.stdout
        iterations = range(num_itr)
    else:
        if os.path.exists(log_dir):
            log = open(log_dir, mode='w', encoding='utf-8')
        else:
            log = open(log_dir, mode='a', encoding='utf-8')
        iterations = tqdm(range(num_itr))
    with torch.no_grad():
        for idx in iterations:
            if cat_model:
                image1, image2, label = test_data[idx]
                image1 = torch.unsqueeze(image1, dim=0)
                image2 = torch.unsqueeze(image2, dim=0)
                output = model(image1.to(device), image2.to(device))
            else:
                image, label = test_data[idx]
                image = torch.unsqueeze(image, dim=0)
                output = model(image.to(device))
            output = torch.squeeze(output).cpu().numpy()
            test_err = np.linalg.norm(output-label, axis=-1)
            x_err = np.linalg.norm(output[0] - label[0])
            y_err = np.linalg.norm(output[1] - label[1])
            z_err = np.linalg.norm(output[2] - label[2])
            total_x_err += x_err
            total_y_err += y_err
            total_z_err += z_err
            print(f"[label] {label}; [output] {output}; [loss] {test_err}", file=log)
            if test_err < 5:
                success += 1
    print(f"Success rate: {success / num_itr}", file=log)
    print(f"Average X error: {total_x_err / num_itr}", file=log)
    print(f"Average Y error: {total_y_err / num_itr}", file=log)
    print(f"Average Z error: {total_z_err / num_itr}", file=log)


def generate_datasets(num_images, file_dir, batch_size, transform):
    np.random.seed(10)
    index_list = list(range(num_images))
    np.random.shuffle(index_list)
    split_num = int(num_images * 0.8)
    train_idx, valid_idx = index_list[:split_num], index_list[split_num:]

    tr_sampler = sampler.SubsetRandomSampler(train_idx)
    val_sampler = sampler.SubsetRandomSampler(valid_idx)

    dataset = TrainDataset(file_dir, transform)
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              drop_last=True, sampler=tr_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size,
                            drop_last=True, sampler=val_sampler)
    print("Size of training set: {}".format(len(train_loader) * batch_size))
    print("Size of validation set: {}".format(len(val_loader) * batch_size))
    return train_loader, val_loader


if __name__ == '__main__':
    p1 = np.array([1, 1, 1])
    p2 = np.array([2, 2, 2])
    print(np.linalg.norm(p2-p1, axis=-1))