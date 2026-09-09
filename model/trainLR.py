import argparse
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import datetime
import time
import matplotlib.pyplot as plt
from torchinfo import summary
import yaml
import json
import sys
import copy

sys.path.append("..")
from lib.utils import (
    MaskedMAELoss,
    print_log,
    seed_everything,
    set_cpu_num,
    CustomJSONEncoder,
)
from model.utils.pltAST import plot_spatial_embedding_tsne, plot_temporal_correlation
from model.utils.serialization import load_adj, load_matrix
from lib.metrics import RMSE_MAE_MAPE
from lib.data_prepareLR import get_dataloaders_from_index_data   # 分SD数据集用这个来划分数据
from LR.ALTP import LRPred

# ! X shape: (B, T, N, C)


@torch.no_grad()
def eval_model(model, valset_loader, criterion):
    model.eval()
    batch_loss_list = []
    for x_batch, y_batch in valset_loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        out_batch = model(x_batch)
        out_batch = SCALER.inverse_transform(out_batch)
        loss = criterion(out_batch, y_batch)
        batch_loss_list.append(loss.item())

    return np.mean(batch_loss_list)


@torch.no_grad()
def predict(model, loader):
    model.eval()
    y = []
    out = []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        out_batch = model(x_batch)
        out_batch = SCALER.inverse_transform(out_batch)

        out_batch = out_batch.cpu().numpy()
        y_batch = y_batch.cpu().numpy()
        out.append(out_batch)
        y.append(y_batch)

    out = np.vstack(out).squeeze()  # (samples, out_steps, num_nodes)
    y = np.vstack(y).squeeze()

    return y, out


def train_one_epoch(
    model, trainset_loader, optimizer, scheduler, criterion, clip_grad, log=None
):
    global cfg, global_iter_count, global_target_length

    model.train()
    batch_loss_list = []
    for x_batch, y_batch in trainset_loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        out_batch = model(x_batch)
        out_batch = SCALER.inverse_transform(out_batch)

        loss = criterion(out_batch, y_batch)
        batch_loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

    epoch_loss = np.mean(batch_loss_list)
    scheduler.step()

    return epoch_loss


def train(
    model,
    trainset_loader,
    valset_loader,
    optimizer,
    scheduler,
    criterion,
    clip_grad=0,
    max_epochs=200,
    early_stop=10,
    verbose=1,
    plot=False,
    log=None,
    save=None,
):
    model = model.to(DEVICE)  # 将模型移动到 GPU

    wait = 0
    min_val_loss = np.inf

    train_loss_list = []
    val_loss_list = []

    for epoch in range(max_epochs):
        train_loss = train_one_epoch(
            model, trainset_loader, optimizer, scheduler, criterion, clip_grad, log=log
        )
        train_loss_list.append(train_loss)

        val_loss = eval_model(model, valset_loader, criterion)
        val_loss_list.append(val_loss)

        if (epoch + 1) % verbose == 0:
            print_log(
                datetime.datetime.now(),
                "Epoch",
                epoch + 1,
                " \tTrain Loss = %.5f" % train_loss,
                "Val Loss = %.5f" % val_loss,
                log=log,
            )

        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= early_stop:
                break
    if save:
        torch.save(best_state_dict, save)
    model.load_state_dict(best_state_dict)
    train_rmse, train_mae, train_mape = RMSE_MAE_MAPE(*predict(model, trainset_loader))
    val_rmse, val_mae, val_mape = RMSE_MAE_MAPE(*predict(model, valset_loader))

    out_str = f"Early stopping at epoch: {epoch+1}\n"
    out_str += f"Best at epoch {best_epoch+1}:\n"
    out_str += "Train Loss = %.5f\n" % train_loss_list[best_epoch]
    out_str += "Train RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
        train_rmse,
        train_mae,
        train_mape,
    )
    out_str += "Val Loss = %.5f\n" % val_loss_list[best_epoch]
    out_str += "Val RMSE = %.5f, MAE = %.5f, MAPE = %.5f" % (
        val_rmse,
        val_mae,
        val_mape,
    )
    print_log(out_str, log=log)

    if plot:
        plt.plot(range(0, epoch + 1), train_loss_list, "-", label="Train Loss")
        plt.plot(range(0, epoch + 1), val_loss_list, "-", label="Val Loss")
        plt.title("Epoch-Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    
    return model


@torch.no_grad()
def test_model(model, testset_loader, log=None):
    model.eval()
    print_log("--------- Test ---------", log=log)

    start = time.time()
    y_true, y_pred = predict(model, testset_loader)
    end = time.time()

    rmse_all, mae_all, mape_all = RMSE_MAE_MAPE(y_true, y_pred)
    out_str = "All Steps RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
        rmse_all,
        mae_all,
        mape_all,
    )
    out_steps = y_pred.shape[1]
    for i in range(out_steps):
        rmse, mae, mape = RMSE_MAE_MAPE(y_true[:, i, :], y_pred[:, i, :])
        out_str += "Step %d RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
            i + 1,
            rmse,
            mae,
            mape,
        )

    print_log(out_str, log=log, end="")
    print_log("Inference time: %.2f s" % (end - start), log=log)
    return y_true, y_pred

@torch.no_grad()
def test_model_new(model, dataloader, log=None):
    model.eval()
    print_log("--------- Test ---------", log=log)

    start = time.time()

    # -------- 全局累计变量 --------
    total_se = 0.0      # squared error
    total_ae = 0.0      # absolute error
    total_ape = 0.0     # absolute percentage error
    total_count = 0     # 有效元素个数（用于 RMSE / MAE）

    total_mape_count = 0  # 专门用于 MAPE（mask 后）

    # -------- per-step --------
    step_se = None
    step_ae = None
    step_ape = None
    step_count = None
    step_mape_count = None

    epsilon = 1e-5

    for x, y in dataloader:
        x = x.to(DEVICE)
        y = y.to(DEVICE).squeeze(dim=-1)

        pred = model(x)
        pred = SCALER.inverse_transform(pred)
        pred = pred.squeeze(dim=-1)

        # shape: (B, T, N)
        if step_se is None:
            T = y.shape[1]
            step_se = torch.zeros(T)
            step_ae = torch.zeros(T)
            step_ape = torch.zeros(T)
            step_count = torch.zeros(T)
            step_mape_count = torch.zeros(T)

        # -------- 误差 --------
        diff = pred - y

        se = diff ** 2
        ae = torch.abs(diff)

        # -------- MAPE（带mask，关键！）--------
        abs_y = torch.abs(y)
        mask = abs_y > epsilon

        ape = torch.zeros_like(ae)
        ape[mask] = ae[mask] / abs_y[mask]

        # -------- 累计（global）--------
        total_se += se.sum().item()
        total_ae += ae.sum().item()
        total_count += se.numel()

        total_ape += ape.sum().item()
        total_mape_count += mask.sum().item()

        # -------- 累计（per-step）--------
        # (B, T, N) → (T,)
        step_se += se.sum(dim=(0, 2)).cpu()
        step_ae += ae.sum(dim=(0, 2)).cpu()
        step_count += torch.tensor(se.shape[0] * se.shape[2]).repeat(T)

        step_ape += ape.sum(dim=(0, 2)).cpu()
        step_mape_count += mask.sum(dim=(0, 2)).cpu()

    # -------- Overall metrics --------
    rmse = np.sqrt(total_se / total_count)
    mae = total_ae / total_count
    mape = total_ape / total_mape_count * 100

    end = time.time()

    out_str = f"All Steps RMSE = {rmse:.5f}, MAE = {mae:.5f}, MAPE = {mape:.5f}\n"

    # -------- per-step metrics --------
    for i in range(T):
        rmse_i = torch.sqrt(step_se[i] / step_count[i]).item()
        mae_i = (step_ae[i] / step_count[i]).item()

        if step_mape_count[i] > 0:
            mape_i = (step_ape[i] / step_mape_count[i]).item() * 100
        else:
            mape_i = float('nan')

        out_str += f"Step {i+1} RMSE = {rmse_i:.5f}, MAE = {mae_i:.5f}, MAPE = {mape_i:.5f}\n"

    print_log(out_str, log=log, end="")
    print_log(f"Inference time: {end - start:.2f} s", log=log)

if __name__ == "__main__":
    # -------------------------- set running environment ------------------------- #

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="pems08")
    parser.add_argument("-g", "--gpu_num", type=int, default=0)
    parser.add_argument("-s", "--seed", type=int, default=1)
    args = parser.parse_args()
    print(args.seed)

    seed_everything(args.seed)
    set_cpu_num(1)

    GPU_ID = args.gpu_num
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPU_ID}"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动检测并使用 GPU

    dataset = args.dataset
    dataset = dataset.upper()
    data_path = f"../data/{dataset}"
    model_name = LRPred.__name__
    # 邻居矩阵
    transition_matrix, _ = load_adj(data_path + "/adj_mx.pkl", "doubletransition")
    transition_matrix = [torch.tensor(i) for i in transition_matrix]
    with open(f"{model_name}.yaml", "r", encoding='utf-8') as f:   # 低秩模型这里有M3
        cfg = yaml.safe_load(f)
    cfg = cfg[dataset]

    # -------------------------------- load model -------------------------------- #

    model = LRPred(transition_matrix=transition_matrix, **cfg["model_args"])    # PEMS有这个-->transition_matrix=transition_matrix   LA和BAY没有
    model = model.to(DEVICE)

    # ------------------------------- make log file ------------------------------ #

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_path = f"../logs/"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log = os.path.join(log_path, f"{model_name}-{dataset}-{now}.log")
    log = open(log, "a")
    log.seek(0)
    log.truncate()

    # ------------------------------- load dataset ------------------------------- #

    print_log(dataset, log=log)
    (
        trainset_loader,
        valset_loader,
        testset_loader,
        SCALER,
    ) = get_dataloaders_from_index_data(
        data_path,
        tod=cfg.get("time_of_day"),
        dow=cfg.get("day_of_week"),
        batch_size=cfg.get("batch_size", 64),
        log=log,
    )
    print_log(log=log)

    # --------------------------- set model saving path -------------------------- #

    save_path = f"../saved_models/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save = os.path.join(save_path, f"{model_name}-{dataset}-{now}.pt")

    # ---------------------- set loss, optimizer, scheduler ---------------------- #

    if dataset in ("METRLA", "PEMSBAY"):
        criterion = MaskedMAELoss()
    elif dataset in ("PEMS03", "PEMS04", "PEMS07", "PEMS08", "PEMSD7M"):
        criterion = nn.HuberLoss()
    else:
        raise ValueError("Unsupported dataset.")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0),
        eps=cfg.get("eps", 1e-8),
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg["milestones"],
        gamma=cfg.get("lr_decay_rate", 0.1),
        verbose=False,
    )

    # --------------------------- print model structure -------------------------- #

    print_log("---------", model_name, "---------", log=log)
    print_log(
        json.dumps(cfg, ensure_ascii=False, indent=4, cls=CustomJSONEncoder), log=log
    )
    print_log(
        summary(
            model,
            [
                cfg["batch_size"],
                cfg["in_steps"],
                cfg["num_nodes"],
                next(iter(trainset_loader))[0].shape[-1],
            ],
            verbose=0,  # avoid print twice
        ),
        log=log,
    )
    print_log(log=log)

    # --------------------------- train and test model --------------------------- #

    print_log(f"Loss: {criterion._get_name()}", log=log)
    print_log(log=log)
    
    model = train(
        model,
        trainset_loader,
        valset_loader,
        optimizer,
        scheduler,
        criterion,
        clip_grad=cfg.get("clip_grad"),
        max_epochs=cfg.get("max_epochs", 200),
        early_stop=cfg.get("early_stop", 10),
        verbose=1,
        log=log,
        save=save,
    )


    # /data/zwf2020/Fastformer/saved_models/LRPred-PEMS08-2026-04-08-01-57-55.pt
    # checkpoint_path = "/data/zwf2020/Fastformer/saved_models/LRPred-PEMS08-2026-04-08-01-57-55.pt"
    # checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))  # 或相应设备
    # model.load_state_dict(checkpoint)
    # model.eval()
    
    print_log(f"Saved Model: {save}", log=log)

    y_true, y_pred = test_model(model, testset_loader, log=log)  # 小数据集还是得用这个，大数据用test_model_new

    log.close()
    # np.savetxt("../vis/08PredLR.csv", y_pred[:, 11, :].T, delimiter=',')
    # np.savetxt("../vis/08True.csv", y_true[:, 11, :].T, delimiter=',')