import numpy as np
import torch
import pickle
import random
import os
import json
import scipy.sparse as sp
import csv
import torch
import torch.nn as nn

def get_adjacent_matrix(distance_file: str, num_nodes: int, id_file: str = None, graph_type="distance", is_text=True) -> np.array:
    """
    :param distance_file: str, path of csv file to save the distances between nodes.
    :param num_nodes: int, number of nodes in the graph
    :param id_file: str, path of txt file to save the order of the nodes.就是排序节点的绝对编号所用到的，这里排好了，不需要
    :param graph_type: str, ["connect", "distance"]，这个就是考不考虑节点之间的距离
    :return:
        np.array(N, N)
    道路领接矩阵构建函数
    """
    A = np.zeros([int(num_nodes), int(num_nodes)])  # 构造全0的邻接矩阵
    if is_text:
        with open(distance_file, "r") as f_d:
            f_d.readline()  # 表头，跳过第一行.
            reader = csv.reader(f_d)  # 读取.csv文件.
            for item in reader:  # 将一行给item组成列表
                if len(item) != 3:  # 长度应为3，不为3则数据有问题，跳过
                    continue
                i, j, distance = int(item[0]), int(item[1]), float(item[2])

                if graph_type == "connect":  # 这个就是将两个节点的权重都设为1，也就相当于不要权重
                    A[i, j], A[j, i] = 1., 1.
                    # A[i, j] = 1.  # 构建有向图
                elif graph_type == "distance":  # 这个是有权重，下面是权重计算方法
                    A[i, j] = 1. / distance
                    A[j, i] = 1. / distance
                else:
                    raise ValueError("graph type is not correct (connect or distance)")
    else:
        A = np.load(distance_file)

    return A


def transform_matrix(matrix, A):
    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0

    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] != 0:
                A[i][j] = 1
                # 对称位置也设置为1
                A[rows - 1 - i][cols - 1 - j] = 1

    return A


def process_graph(graph_data):  # 这个就是在原始的邻接矩阵之上，再次变换，也就是\hat A = D_{-1/2}*A*D_{-1/2}
    N = graph_data.size(0)  # 获得节点的个数
    matrix_i = torch.eye(N, dtype=torch.float, device=graph_data.device)  # 定义[N, N]的单位矩阵
    graph_data += matrix_i  # [N, N]  ,就是 A+I

    degree_matrix = torch.sum(graph_data, dim=1, keepdim=False)  # [N],计算度矩阵，塌陷成向量，其实就是将上面的A+I每行相加
    degree_matrix = degree_matrix.pow(-1)  # 计算度矩阵的逆，若为0，-1次方可能计算结果为无穷大的数
    degree_matrix[degree_matrix == float("inf")] = 0.  # 让无穷大的数为0

    degree_matrix = torch.diag(degree_matrix)  # 转换成对角矩阵

    return torch.mm(degree_matrix, graph_data)  # 返回 \hat A=D^(-1) * A ,这个等价于\hat A = D_{-1/2}*A*D_{-1/2}



def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()


def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


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


def load_adj(pkl_filename):
    # if any(field in pkl_filename for field in ["PEMSBAY", "METRLA"]):   # or只能处理两个变量
    _, _, adj_mx = load_pickle(pkl_filename)  # 08只返回一个adj_mx,bay和mertla返回三个
    # if any(field in pkl_filename for field in ["03", "04", "08", "07"]):
    #     adj_mx = load_pickle(pkl_filename)
    # adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]

    return adj_mx


def calculate_transition_matrix(adj: np.ndarray) -> np.matrix:
    """adj-->[N,N]
    Calculate the transition matrix `P` proposed in DCRNN and Graph WaveNet.
    P = D^{-1}A = A/rowsum(A)
    Args:
        adj (np.ndarray): Adjacent matrix A
    Returns:
        np.matrix: Transition matrix P
    """

    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(row_sum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    prob_matrix = d_mat.dot(adj).astype(np.float32).todense()
    return prob_matrix


class StandardScaler:
    """
    Standard the input
    https://github.com/nnzhan/Graph-WaveNet/blob/master/util.py
    """

    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit_transform(self, data):
        self.mean = data.mean()
        self.std = data.std()

        return (data - self.mean) / self.std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def masked_mae_loss(preds, labels, null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mae(preds, labels, null_val):
    if not isinstance(preds, torch.Tensor):
        preds = torch.from_numpy(preds)
        labels = torch.from_numpy(labels)
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

class MaskedMAELoss:
    def _get_name(self):
        return self.__class__.__name__

    def __call__(self, preds, labels, null_val=0.0):  # 替换为np.nan torch.tensor(0.0)， 原始是0.0
        return masked_mae_loss(preds, labels, null_val)   # 这里做了改变masked_mae替换了masked_mae_loss
        
class MaskedMAELossMiss(nn.Module):
    def __init__(self, eps=1e-8):
        super(MaskedMAELoss, self).__init__()
        self.eps = eps

    def forward(self, out_batch, y_batch):
        """
        out_batch: 预测值，形状 [batch, ...]
        y_batch:   真实标签，形状与 out_batch 相同
        """
        # 生成掩码：非零位置为 1，零位置为 0
        mask = (y_batch != 0).float()
        
        # 计算绝对误差
        abs_error = torch.abs(out_batch - y_batch)
        
        # 应用掩码并求和
        masked_error = abs_error * mask
        loss = masked_error.sum() / (mask.sum() + self.eps)
        
        return loss


def print_log(*values, log=None, end="\n"):
    print(*values, end=end)
    if log:
        if isinstance(log, str):
            log = open(log, "a")
        print(*values, file=log, end=end)
        log.flush()


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def set_cpu_num(cpu_num: int):
    os.environ["OMP_NUM_THREADS"] = str(cpu_num)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
    os.environ["MKL_NUM_THREADS"] = str(cpu_num)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
    torch.set_num_threads(cpu_num)


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return f"Shape: {obj.shape}"
        elif isinstance(obj, torch.device):
            return str(obj)
        else:
            return super(CustomJSONEncoder, self).default(obj)


def vrange(starts, stops):
    """Create ranges of integers for multiple start/stop

    Parameters:
        starts (1-D array_like): starts for each range
        stops (1-D array_like): stops for each range (same shape as starts)
        
        Lengths of each range should be equal.

    Returns:
        numpy.ndarray: 2d array for each range
        
    For example:

        >>> starts = [1, 2, 3, 4]
        >>> stops  = [4, 5, 6, 7]
        >>> vrange(starts, stops)
        array([[1, 2, 3],
               [2, 3, 4],
               [3, 4, 5],
               [4, 5, 6]])

    Ref: https://codereview.stackexchange.com/questions/83018/vectorized-numpy-version-of-arange-with-multiple-start-stop
    """
    stops = np.asarray(stops)
    l = stops - starts  # Lengths of each range. Should be equal, e.g. [12, 12, 12, ...]
    assert l.min() == l.max(), "Lengths of each range should be equal."
    indices = np.repeat(stops - l.cumsum(), l) + np.arange(l.sum())
    return indices.reshape(-1, l[0])


def print_model_params(model):
    param_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print("%-40s\t%-30s\t%-30s" % (name, list(param.shape), param.numel()))
            param_count += param.numel()
    print("%-40s\t%-30s" % ("Total trainable params", param_count))


if __name__=='__main__':
    path = '../data/PEMSBAY/adj.pkl'
    # out = get_adjacent_matrix(path, 325, is_text=False)
    out = np.load('../data/PEMSBAY/adj.npy')