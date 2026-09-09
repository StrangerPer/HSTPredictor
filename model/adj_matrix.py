import csv
import numpy as np
import torch
import scipy.sparse as sp
import os
import pickle


def get_adjacent_matrix(distance_file: str, num_nodes: int, id_file: str = None, graph_type="distance") -> np.array:
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

    if id_file:  # 就是给节点排序的绝对文件，这里是None，则表示不需要
        with open(id_file, "r") as f_id:
            # 将绝对编号用enumerate()函数打包成一个索引序列，然后用node_id这个绝对编号做key，用idx这个索引做value
            node_id_dict = {int(node_id): idx for idx, node_id in enumerate(f_id.read().strip().split("\n"))}

            with open(distance_file, "r") as f_d:
                f_d.readline()  # 表头，跳过第一行.
                reader = csv.reader(f_d)  # 读取.csv文件.
                for item in reader:  # 将一行给item组成列表
                    if len(item) != 3:  # 长度应为3，不为3则数据有问题，跳过
                        continue
                    i, j, distance = int(item[0]), int(item[1]), float(item[2])  # 节点i，节点j，距离distance
                    if graph_type == "connect":  # 这个就是将两个节点的权重都设为1，也就相当于不要权重
                        A[node_id_dict[i], node_id_dict[j]] = 1.
                        A[node_id_dict[j], node_id_dict[i]] = 1.
                    elif graph_type == "distance":  # 这个是有权重，下面是权重计算方法
                        A[node_id_dict[i], node_id_dict[j]] = 1. / distance
                        A[node_id_dict[j], node_id_dict[i]] = 1. / distance
                    else:
                        raise ValueError("graph type is not correct (connect or distance)")
        return A

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

    return A       # 返回的是一个Numpy

# 计算转移矩阵
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


def scaled_laplacian(A):
    n = A.shape[0]
    d = np.sum(A, axis=1)
    L = np.diag(d) - A
    for i in range(n):
        for j in range(n):
            if d[i] > 0 and d[j] > 0:
                L[i, j] /= np.sqrt(d[i] * d[j])
    lam = np.linalg.eigvals(L).max().real
    return 2 * L / lam - np.eye(n)        # 输出为np类型


def process_graph(graph_data):  # 这个就是在原始的邻接矩阵之上，再次变换，也就是\hat A = D_{-1/2}*A*D_{-1/2}
    N = graph_data.size(0)  # 获得节点的个数
    matrix_i = torch.eye(N, dtype=torch.float, device=graph_data.device)  # 定义[N, N]的单位矩阵
    graph_data += matrix_i  # [N, N]  ,就是 A+I

    degree_matrix = torch.sum(graph_data, dim=1, keepdim=False)  # [N],计算度矩阵，塌陷成向量，其实就是将上面的A+I每行相加
    degree_matrix = degree_matrix.pow(-1)  # 计算度矩阵的逆，若为0，-1次方可能计算结果为无穷大的数
    degree_matrix[degree_matrix == float("inf")] = 0.  # 让无穷大的数为0

    degree_matrix = torch.diag(degree_matrix)  # 转换成对角矩阵

    return torch.mm(degree_matrix, graph_data)  # 返回 \hat A=D^(-1) * A ,这个等价于\hat A = D_{-1/2}*A*D_{-1/2}


def graphNorm(graph_data):
    N = graph_data.size(0)
    I = torch.eye(N, dtype=torch.float, device=graph_data.device)  # 定义[N, N]的单位矩阵
    graph_data += I  # [N, N]  ,就是 A+I
    # 计算度矩阵
    degree_matrix = torch.sum(graph_data, dim=1, keepdim=False)  # [N],计算度矩阵，塌陷成向量，其实就是将上面的A+I每行相加
    degree_matrix = torch.pow(degree_matrix, -0.5)
    degree_matrix[degree_matrix == float("inf")] = 0.  # 让无穷大的数为0
    degree_matrix = torch.diag(degree_matrix)  # 转换成对角矩阵
    normalized_graph = torch.mm(degree_matrix, torch.mm(graph_data, degree_matrix))
    return graph_data, normalized_graph

def graph2Z(graph_data):
    # # D0.5 A D-0.5
    N = graph_data.size(0)
    I = torch.eye(N, dtype=torch.float, device=graph_data.device)  # 定义[N, N]的单位矩阵
    graph_data += I  # [N, N]  ,就是 A+I

    degree_matrix = torch.sum(graph_data, dim=1, keepdim=False)
    degree_matrix_right = torch.pow(degree_matrix, -0.5)
    degree_matrix_right[degree_matrix_right == float("inf")] = 0.  # 让无穷大的数为0
    degree_matrix_left = torch.pow(degree_matrix, 0.5)
    # 两个都要转换为对角矩阵
    degree_matrix_left = torch.diag(degree_matrix_left)
    degree_matrix_right = torch.diag(degree_matrix_right)
    # D(0.5)AD(-0.5)
    norm_adjZ = torch.mm(degree_matrix_left, torch.mm(graph_data, degree_matrix_right))
    return norm_adjZ


def get_degree(edge_list):
    deg = torch.sum(edge_list, dim=1, keepdim=False)
    return deg

def normalize_adj(edge_list):
    # D-0.5 A D-0.5
    deg = get_degree(edge_list)
    row, col = edge_list
    deg_inv_sqrt = torch.pow(deg.to(torch.float), -0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
    weight = torch.ones(edge_list.size(1))
    v = deg_inv_sqrt[row] * weight * deg_inv_sqrt[col]
    norm_adj = torch.sparse.FloatTensor(edge_list, v)
    return norm_adj

def normalize_adj2Z(edge_list):
    # D0.5 A D-0.5
    deg = get_degree(edge_list)
    row, col = edge_list
    deg_inv_sqrt_right = torch.pow(deg.to(torch.float), -0.5)
    deg_inv_sqrt_right[deg_inv_sqrt_right == float('inf')] = 0.0
    deg_inv_sqrt_left = torch.pow(deg.to(torch.float), 0.5)
    weight = torch.ones(edge_list.size(1))
    v = deg_inv_sqrt_left[row] * weight * deg_inv_sqrt_right[col]
    norm_adj = torch.sparse.FloatTensor(edge_list, v)
    return norm_adj


def get_adjacency_matrix03(distance_df_filename: str, num_of_vertices: int, id_filename: str = None) -> tuple:
    """Generate adjacency matrix.

    Args:
        distance_df_filename (str): path of the csv file contains edges information
        num_of_vertices (int): number of vertices
        id_filename (str, optional): id filename. Defaults to None.

    Returns:
        tuple: two adjacency matrix.
            np.array: connectivity-based adjacency matrix A (A[i, j]=0 or A[i, j]=1)
            np.array: distance-based adjacency matrix A
    """

    if "npy" in distance_df_filename:
        adj_mx = np.load(distance_df_filename)
        return adj_mx, None
    else:
        adjacency_matrix_connectivity = np.zeros((int(num_of_vertices), int(
            num_of_vertices)), dtype=np.float32)
        adjacency_matrix_distance = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                                             dtype=np.float32)
        if id_filename:
            # the id in the distance file does not start from 0, so it needs to be remapped
            with open(id_filename, "r") as f:
                id_dict = {int(i): idx for idx, i in enumerate(
                    f.read().strip().split("\n"))}  # map node idx to 0-based index (start from 0)
                # 将他们映射为0-357之间来
            with open(distance_df_filename, "r") as f:
                f.readline()  # omit the first line
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    adjacency_matrix_connectivity[id_dict[i], id_dict[j]] = 1
                    adjacency_matrix_connectivity[id_dict[j], id_dict[i]] = 1
                    adjacency_matrix_distance[id_dict[i],
                                              id_dict[j]] = distance
                    adjacency_matrix_distance[id_dict[j],
                                              id_dict[i]] = distance
            return adjacency_matrix_connectivity, adjacency_matrix_distance
        else:
            # ids in distance file start from 0
            with open(distance_df_filename, "r") as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    adjacency_matrix_connectivity[i, j] = 1
                    adjacency_matrix_connectivity[j, i] = 1
                    adjacency_matrix_distance[i, j] = distance
                    adjacency_matrix_distance[j, i] = distance
            return adjacency_matrix_connectivity, adjacency_matrix_distance


def generate_adj_pems03():
    path = "E:\GinAR-main (2)\GinAR-main\data\PEMS03\PEMS03.csv"
    distance_df_filename, num_of_vertices = path, 358
    if os.path.exists(distance_df_filename.split(".", maxsplit=1)[0] + ".txt"):
        id_filename = distance_df_filename.split(".", maxsplit=1)[0] + ".txt"
    else:
        id_filename = None
    adj_mx, distance_mx = get_adjacency_matrix03(
        distance_df_filename, num_of_vertices, id_filename=id_filename)
    # the self loop is missing
    # add_self_loop = False
    # if add_self_loop:
    #     print("adding self loop to adjacency matrices.")
    #     adj_mx = adj_mx + np.identity(adj_mx.shape[0])
    #     distance_mx = distance_mx + np.identity(distance_mx.shape[0])
    # else:
    #     print("kindly note that there is no self loop in adjacency matrices.")
    # with open("E:/GinAR-main (2)/GinAR-main/data\PEMS03/adj_PEMS03.pkl", "wb") as f:
    #     pickle.dump(adj_mx, f)
    # with open("E:/GinAR-main (2)/GinAR-main\data\PEMS03/adj_PEMS03_distance.pkl", "wb") as f:
    #     pickle.dump(distance_mx, f)
    return adj_mx



if __name__ == '__main__':
    out = generate_adj_pems03()
    print(out[2])
