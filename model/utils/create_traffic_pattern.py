import os
import numpy as np
import pickle
from tslearn.clustering import KShape
from tqdm import tqdm

class PatternGenerator:
    def __init__(self, config):
        self.config = config
        self.data_dir = config.get('data_dir', './data')
        self.time_intervals = config.get('time_intervals', 300)
        self.cand_key_days = config.get("cand_key_days", 14)
        self.s_attn_size = config.get("s_attn_size", 5)
        self.n_cluster = config.get("n_cluster", 16)
        self.cluster_max_iter = config.get("cluster_max_iter", 5)
        self.cluster_method = config.get("cluster_method", "kshape")
        self.sample_ratio = config.get("sample_ratio", 0.1)  # 采样比例
        self.target_dim = 0  
        
        self.points_per_hour = 3600 // self.time_intervals
        self.points_per_day = 24 * self.points_per_hour
        
        self._generate_and_save_pattern()

    def _load_traffic_data(self):
        data = np.load(os.path.join(self.data_dir, "data.npz"))["data"]
        index = np.load(os.path.join(self.data_dir, "index.npz"))
        return data[index["train"], :, self.target_dim]  

    def _generate_pattern_keys(self, traffic_data):
        cand_key_steps = self.cand_key_days * self.points_per_day
        traffic_data = traffic_data[:min(len(traffic_data), cand_key_steps)]
        
        # 构造滑动窗口
        patterns = []
        for i in range(len(traffic_data) - self.s_attn_size + 1):
            patterns.append(traffic_data[i:i+self.s_attn_size])
        
        pattern_array = np.array(patterns).swapaxes(1, 2)  # (n_samples, seq_len, 1)
        pattern_array = pattern_array.reshape(-1, self.s_attn_size, 1) 

        # ---------- 采样版聚类 ----------
        n_samples = pattern_array.shape[0]
        sample_size = max(1, int(n_samples * self.sample_ratio))
        np.random.seed(42)
        sample_idx = np.random.choice(n_samples, sample_size, replace=False)
        sampled_data = pattern_array[sample_idx]

        if self.cluster_method == "kshape":
            km = KShape(n_clusters=self.n_cluster, max_iter=self.cluster_max_iter, random_state=42)
        else:
            from tslearn.clustering import TimeSeriesKMeans
            km = TimeSeriesKMeans(n_clusters=self.n_cluster, metric="softdtw", 
                                  max_iter=self.cluster_max_iter, random_state=42)

        print(f"Clustering {sample_size}/{n_samples} sampled patterns...")
        km.fit(sampled_data)
        cluster_centers = km.cluster_centers_

        # ---------- 向量化映射全量数据到最近簇 ----------
        # 1. pattern_array: (n_samples, seq_len, dim)
        # 2. cluster_centers: (n_clusters, seq_len, dim)
        # 通过广播计算欧氏距离
        # distance[i, j] = || pattern_array[i] - cluster_centers[j] ||^2
        diff = pattern_array[:, None, :, :] - cluster_centers[None, :, :, :]  # (n_samples, n_clusters, seq_len, dim)
        distances = np.sum(diff**2, axis=(2, 3))  # (n_samples, n_clusters)
        labels = np.argmin(distances, axis=1)      # (n_samples,)

        print("All patterns assigned to nearest cluster center (vectorized).")
        return cluster_centers, labels

    def _generate_and_save_pattern(self):
        pkl_path = os.path.join(self.data_dir, 'pattern_key.pkl')
        
        if os.path.exists(pkl_path):
            print(f"Pattern keys already exist at {pkl_path}")
            return
        
        print("Generating pattern keys...")
        traffic_data = self._load_traffic_data()
        pattern_keys, labels = self._generate_pattern_keys(traffic_data)
        
        with open(pkl_path, 'wb') as f:
            pickle.dump({"cluster_centers": pattern_keys, "labels": labels}, f)
        print(f"Pattern keys saved to {pkl_path}")


if __name__ == "__main__":
    config = {
        'data_dir': '/data/zwf2020/STHDformer/data/METRLA/',
        'time_intervals': 300,
        'cand_key_days': 14,
        's_attn_size': 3,
        'n_cluster': 16,
        'cluster_max_iter': 5,
        'cluster_method': "kshape",
        'sample_ratio': 0.2  # 只用 10% 数据做聚类
    }
    
    generator = PatternGenerator(config)
