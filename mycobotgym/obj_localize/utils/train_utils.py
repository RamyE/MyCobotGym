import sys
import torch
import numpy as np

from tqdm import tqdm
from torch.utils.data import sampler
from torch.utils.data import DataLoader
from mycobotgym.obj_localize.vision_model.dataset import TrainDataset


def train(model, optimizer, loss_function, tr_loader, device, epoch, epochs):
    model.train()
    train_bar = tqdm(tr_loader, file=sys.stdout)
    num_itr = 0
    train_acc_loss = 0
    for step, data in enumerate(train_bar):
        train_images, train_labels = data
        optimizer.zero_grad()
        train_outputs = model(train_images.to(device))
        train_loss = loss_function(train_outputs, train_labels.to(device))
        train_loss.backward()
        optimizer.step()
        train_acc_loss += train_loss.item()
        num_itr += 1
        train_bar.desc = "Epoch[{}/{}] train loss:{}".format(epoch + 1, epochs, train_loss)
    return train_loss.item(), train_acc_loss / num_itr


def validation(model, loss_function, val_loader, device, epoch, epochs):
    model.eval()
    with torch.no_grad():
        val_bar = tqdm(val_loader, file=sys.stdout)
        num_itr = 0
        val_acc_loss = 0
        for val_data in val_bar:
            val_images, val_labels = val_data
            val_outputs = model(val_images.to(device))
            val_loss = loss_function(val_outputs, val_labels.to(device))
            val_acc_loss += val_loss.item()
            num_itr += 1
            val_bar.desc = "Epoch[{}/{}] val loss:{}".format(epoch + 1, epochs, val_loss)
    return val_loss.item(), val_acc_loss / num_itr


def run_test(model, device, test_data, num_itr=None):
    model.eval()
    success = 0
    if num_itr is None:
        num_itr = len(test_data)
    with torch.no_grad():
        for idx in range(num_itr):
            image, label = test_data[idx]
            image = torch.unsqueeze(image, dim=0)
            output = model(image.to(device))
            output = torch.squeeze(output).cpu().numpy()
            test_loss = np.linalg.norm(output-label, axis=-1)
            print(f"label: {label}; output: {output}; loss: {test_loss}")
            if test_loss < 5:
                success += 1
    print(f"Success rate: {success / num_itr}")


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
