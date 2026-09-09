import torch
import numpy as np
import os
from .utils import print_log, StandardScaler, vrange

# ! X shape: (B, T, N, C)
def generate_mask_for(feature, missing_probability):
    # 生成与 feature 相同形状的均匀分布随机数矩阵
    random_vals = np.random.uniform(0.0, 1.0, size=feature.shape)
    # 大于 missing_probability 的位置为 1，否则为 0
    mask = (random_vals > missing_probability).astype(np.float32)
    return mask


def get_dataloaders_from_index_data(
    data_dir, missing_rate, tod=False, dow=False, dom=False, batch_size=64, log=None
):
    data = np.load(os.path.join(data_dir, "data"+".npz"))["data"].astype(np.float32)

    features = [0]
    mask = generate_mask_for(data[..., :1], missing_rate)
    
    if tod:
        features.append(1)
    if dow:
        features.append(2)
    # if dom:
    #     features.append(3)
    data = data[..., features]
    index = np.load(os.path.join(data_dir, "index"+".npz"))

    train_index = index["train"]  # (num_samples, 3)
    print(train_index.shape)
    val_index = index["val"]
    test_index = index["test"]

    x_train_index = vrange(train_index[:, 0], train_index[:, 1])
    y_train_index = vrange(train_index[:, 1], train_index[:, 2])
    x_val_index = vrange(val_index[:, 0], val_index[:, 1])
    y_val_index = vrange(val_index[:, 1], val_index[:, 2])
    x_test_index = vrange(test_index[:, 0], test_index[:, 1])
    y_test_index = vrange(test_index[:, 1], test_index[:, 2])
    
    x_train = data[x_train_index]
    y_train = data[y_train_index][..., :1]
    x_val = data[x_val_index]
    y_val = data[y_val_index][..., :1]
    x_test = data[x_test_index]
    y_test = data[y_test_index][..., :1]

    scaler = StandardScaler(mean=x_train[..., 0].mean(), std=x_train[..., 0].std())

    x_train[..., 0] = scaler.transform(x_train[..., 0])
    x_val[..., 0] = scaler.transform(x_val[..., 0])
    x_test[..., 0] = scaler.transform(x_test[..., 0])
    # 先进行归一化，然后再进行后续操作

    x_train_mask = mask[x_train_index]
    y_train_mask = mask[y_train_index]
    x_train = data[x_train_index] * x_train_mask
    y_train = data[y_train_index][..., :1] * y_train_mask

    x_val_mask = mask[x_val_index]
    y_val_mask = mask[y_val_index]
    x_val = data[x_val_index] * x_val_mask
    y_val = data[y_val_index][..., :1] * y_val_mask
    
    x_test_mask = mask[x_test_index]
    y_test_mask = mask[y_test_index]
    x_test = data[x_test_index] * x_test_mask
    y_test = data[y_test_index][..., :1]

    print_log(f"Trainset:\tx-{x_train.shape}\ty-{y_train.shape}", log=log)
    print_log(f"Valset:  \tx-{x_val.shape}  \ty-{y_val.shape}", log=log)
    print_log(f"Testset:\tx-{x_test.shape}\ty-{y_test.shape}", log=log)

    trainset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_train), torch.FloatTensor(y_train), 
    )
    valset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_val), torch.FloatTensor(y_val)
    )
    testset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_test), torch.FloatTensor(y_test)
    )

    trainset_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )
    valset_loader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=False
    )
    testset_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )

    return trainset_loader, valset_loader, testset_loader, scaler
