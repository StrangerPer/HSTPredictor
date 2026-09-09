import numpy as np
from fastdtw import fastdtw
from joblib import Parallel, delayed
import os
import pickle
from tqdm import tqdm
import networkx as nx
from gensim.models import Word2Vec
import random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity  

def vrange(starts, stops):
    stops = np.asarray(stops)
    l = stops - starts
    assert l.min() == l.max(), "Lengths of each range should be equal."
    indices = np.repeat(stops - l.cumsum(), l) + np.arange(l.sum())
    return indices.reshape(-1, l[0])

def calculate_raw_dtw_matrix(data_dir='./data/PEMS08/', radius=6, n_jobs=-1):
    try:
        data = np.load(os.path.join(data_dir, "data.npz"))["data"].astype(np.float32)
        index = np.load(os.path.join(data_dir, "index.npz"))
        train_index = index["train"]
    except Exception as e:
        raise ValueError(f"Failed to load the data: {str(e)}")
    
    x_train_index = vrange(train_index[:, 0], train_index[:, 1])
    train_data = data[x_train_index]
    train_data = train_data.transpose(0, 2, 1, 3)
    points_per_day = 24 * 12
    total_time_points = train_data.shape[0]
    num_days = total_time_points // points_per_day
    complete_days_data = train_data[:num_days * points_per_day]
    
    daily_data = complete_days_data.reshape(
        num_days, points_per_day, train_data.shape[1], train_data.shape[2], train_data.shape[3]
    )
    data_mean = np.mean(daily_data, axis=0)

    cache_path = os.path.join(data_dir, 'dtw_matrix.pkl')
    
    if not os.path.exists(cache_path):
        num_nodes = data_mean.shape[1]
        dtw_matrix = np.zeros((num_nodes, num_nodes))
 
        def compute_dtw(i, j):
            if i <= j:
                dist, _ = fastdtw(data_mean[:, i, 0], data_mean[:, j, 0], radius=radius)
                return i, j, dist
            return i, j, 0

        pairs = [(i, j) for i in range(num_nodes) for j in range(num_nodes)]
        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_dtw)(i, j) for i, j in tqdm(pairs, desc="The calculation progress of DTW"))

        for i, j, dist in results:
            dtw_matrix[i, j] = dist
        dtw_matrix = np.maximum(dtw_matrix, dtw_matrix.T)

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(dtw_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            os.remove(cache_path) if os.path.exists(cache_path) else None
            raise IOError(f"Failed to save the file: {str(e)}")

    try:
        with open(cache_path, 'rb') as f:
            dtw_matrix = pickle.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load cache file: {str(e)}")

    return dtw_matrix

def build_graph_from_dtw(dtw_matrix, threshold_percentile=90, max_edges_per_node=None, use_dtw_as_weight=True):
    num_nodes = dtw_matrix.shape[0]
    
    if use_dtw_as_weight:
        spatial_semantics = 1 / (dtw_matrix + 1e-8)
    else:
        dtw_min = np.min(dtw_matrix)
        dtw_max = np.max(dtw_matrix)
        spatial_semantics = 1 - (dtw_matrix - dtw_min) / (dtw_max - dtw_min + 1e-8)
    
    np.fill_diagonal(spatial_semantics, 0)
    
    threshold = np.percentile(spatial_semantics[spatial_semantics > 0], threshold_percentile)
    
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    
    for i in range(num_nodes):
        neighbors = [(j, spatial_semantics[i, j]) for j in range(num_nodes) if j != i]
        neighbors.sort(key=lambda x: x[1], reverse=True)
        if max_edges_per_node is not None:
            neighbors = neighbors[:max_edges_per_node]
        
        for j, sim in neighbors:
            if sim >= threshold:
                G.add_edge(i, j, weight=sim)
    
    return G

def create_alias_table(area_ratio):
    l = len(area_ratio)
    accept, alias = [0] * l, [0] * l
    small, large = [], []
    
    for i, prob in enumerate(area_ratio):
        if prob < 1.0:
            small.append(i)
        else:
            large.append(i)
    
    while small and large:
        small_idx, large_idx = small.pop(), large.pop()
        accept[small_idx] = area_ratio[small_idx]
        alias[small_idx] = large_idx
        area_ratio[large_idx] = area_ratio[large_idx] - (1 - area_ratio[small_idx])
        
        if area_ratio[large_idx] < 1.0:
            small.append(large_idx)
        else:
            large.append(large_idx)
    
    while large:
        large_idx = large.pop()
        accept[large_idx] = 1
    while small:
        small_idx = small.pop()
        accept[small_idx] = 1
    
    return accept, alias

def alias_sample(accept, alias):
    N = len(accept)
    i = int(np.random.random() * N)
    r = np.random.random()
    if r < accept[i]:
        return i
    else:
        return alias[i]

class Node2Vec:
    def __init__(self, graph, walk_length=80, num_walks=10, workers=4):
        self.G = graph
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.workers = workers
        self.alias_nodes = {}
        
        self._preprocess_transition_probs()
    
    def _preprocess_transition_probs(self):
        for node in self.G.nodes():
            neighbors = list(self.G.neighbors(node))
            if not neighbors:
                continue
            
            weights = [self.G[node][nbr].get('weight', 1.0) for nbr in neighbors]
            weight_sum = sum(weights)
            probs = [w / weight_sum for w in weights]
            self.alias_nodes[node] = create_alias_table(probs)
    
    def _unbiased_walk(self, start_node):
        walk = [start_node]
        
        while len(walk) < self.walk_length:
            cur_node = walk[-1]
            neighbors = list(self.G.neighbors(cur_node))
            
            if len(neighbors) == 0:
                break
            
            next_node = neighbors[alias_sample(*self.alias_nodes[cur_node])]
            walk.append(next_node)
        
        return walk
    
    def simulate_walks(self):
        walks = []
        nodes = list(self.G.nodes())
        
        for _ in range(self.num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self._unbiased_walk(node))
        
        return walks
    
    def learn_embeddings(self, dimensions=128, window_size=10, min_count=1, sg=1):
        walks = self.simulate_walks()
        walks = [list(map(str, walk)) for walk in walks]
        
        model = Word2Vec(
            walks,
            vector_size=dimensions,
            window=window_size,
            min_count=min_count,
            sg=sg,
            workers=self.workers,
            epochs=10
        )
        
        return model

def get_node2vec_embeddings(dtw_matrix, threshold_percentile=85, max_edges_per_node=20, 
                          walk_length=80, num_walks=10, dimensions=64):
    G = build_graph_from_dtw(dtw_matrix, threshold_percentile, max_edges_per_node, use_dtw_as_weight=True)
    
    node2vec = Node2Vec(G, walk_length=walk_length, num_walks=num_walks)
    node2vec_model = node2vec.learn_embeddings(dimensions=dimensions)
    
    nodes = list(node2vec_model.wv.index_to_key)
    embedding_matrix = np.array([node2vec_model.wv.get_vector(str(node)) for node in nodes])
    spatial_semantics = cosine_similarity(embedding_matrix)
    np.fill_diagonal(spatial_semantics, 0)
    
    return G, embedding_matrix, node2vec_model, spatial_semantics

if __name__ == "__main__":
    dtw_matrix = calculate_raw_dtw_matrix(data_dir='/data/zwf2020/Fastformer/data/METRLA/')
    G, node_embeddings, node2vec_model, spatial_semantics = get_node2vec_embeddings(
        dtw_matrix, 
        threshold_percentile=85, 
        max_edges_per_node=20,
        walk_length=80, 
        num_walks=10, 
        dimensions=64
    )
    
    result_dir = '/data/zwf2020/Fastformer/data/METRLA/'
    os.makedirs(result_dir, exist_ok=True)

    similarity_pkl_path = os.path.join(result_dir, 'spatial_semantics.pkl')
    with open(similarity_pkl_path, 'wb') as f:
        pickle.dump(spatial_semantics, f, protocol=pickle.HIGHEST_PROTOCOL)