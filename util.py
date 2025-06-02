import numpy as np
import os
import scipy.sparse as sp
import torch
import pickle

def load_graph_data(pkl_filename):
    adj_mx = load_pickle(pkl_filename)
    return adj_mx

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        self.batch_size = batch_size  # 64
        self.current_ind = 0
        """
        This operation adds extra samples (rows) to the bottom of your data array, 
        usually to make the number of samples a multiple of your batch size for training.
        """
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)  # (18num_padding, 12, 250, 3)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)  # (18num_padding, 12, 250, 1)
            xs = np.concatenate([xs, x_padding], axis=0)  # (2624, 12, 250, 3)
            ys = np.concatenate([ys, y_padding], axis=0)  # (2624, 12, 250, 1)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind:end_ind, ...]
                y_i = self.ys[start_ind:end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def load_dataset(dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None):
    data = {}
    for category in ["train", "val", "test"]:
        cat_data = np.load(os.path.join(dataset_dir, category + ".npz"))
        data["x_" + category] = cat_data["x"]  # (2606, 12, 250, 3) 2606个样本，时间，节点，通道/特征数量 (流量值, day, week) day[0,1] week[0]  np.max(data['x_train'][..., 2]) to access the third feature (index 2 in the fourth dimension) and find its maximum value
        data["y_" + category] = cat_data["y"]  # train:2606 val:870 test:869  train(2606, 12, 250, 1)  val:(870, 12, 250, 1)  
    scaler = StandardScaler(
        mean=data["x_train"][..., 0].mean(), std=data["x_train"][..., 0].std()  # 提取训练集所有样本、所有时间步、所有站点的 第0个特征
    )
    # Data format
    for category in ["train", "val", "test"]: # 把第一个特征标准化 [0,1]
        data["x_" + category][..., 0] = scaler.transform(data["x_" + category][..., 0])

    print("Perform shuffle on the dataset")
    random_train = torch.arange(int(data["x_train"].shape[0]))  # tensor([0,1,2, ..., 2603,2604,2605])
    random_train = torch.randperm(random_train.size(0))  # 打乱
    """
    This line shuffles the order of your samples (the first dimension of x_train) according to random_train.
    All other dimensions (timesteps, locations, features) remain unchanged.
    """
    data["x_train"] = data["x_train"][random_train, ...]  # (2606, 12, 250, 3)
    data["y_train"] = data["y_train"][random_train, ...]  # (2606, 12, 250, 1)

    random_val = torch.arange(int(data["x_val"].shape[0]))
    random_val = torch.randperm(random_val.size(0))
    data["x_val"] = data["x_val"][random_val, ...]
    data["y_val"] = data["y_val"][random_val, ...]

    # random_test = torch.arange(int(data['x_test'].shape[0]))
    # random_test = torch.randperm(random_test.size(0))
    # data['x_test'] =  data['x_test'][random_test,...]
    # data['y_test'] =  data['y_test'][random_test,...]

    data["train_loader"] = DataLoader(data["x_train"], data["y_train"], batch_size)
    data["val_loader"] = DataLoader(data["x_val"], data["y_val"], valid_batch_size)
    data["test_loader"] = DataLoader(data["x_test"], data["y_test"], test_batch_size)
    data["scaler"] = scaler

    return data

def MAE_torch(pred, true, mask_value=None):  # The mask is used to filter out (ignore) values in your tensors where the target (true) is less than or equal to mask_value, so the MAE is only calculated over "real" or "valid" data points.
    if mask_value != None:  # To ignore invalid or missing values in your data (for example, if missing values are represented as -999).
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(true - pred))


def MAPE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(torch.div((true - pred), true)))


def RMSE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.sqrt(torch.mean((pred - true) ** 2))


def WMAPE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    loss = torch.sum(torch.abs(pred - true)) / torch.sum(torch.abs(true))
    return loss

def metric(pred, real):
    mae = MAE_torch(pred, real, 0).item()
    mape = MAPE_torch(pred, real,0).item()
    wmape = WMAPE_torch(pred, real, 0).item()
    rmse = RMSE_torch(pred, real, 0).item()
    return mae, mape, rmse, wmape


