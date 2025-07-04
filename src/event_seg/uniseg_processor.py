import numpy as np
import networkx as nx
import math
from tqdm import tqdm
from config import Config
from feature_extractor import FeatureExtractor
from graph_operations import graph_propagation  # 确保正确导入
from boundary_detection import detect_boundaries  # 确保正确导入

class UniSegProcessor:
    def __init__(self):
        self.extractor = FeatureExtractor()
        
    def build_dynamic_graph(self, features, fps):
        n = len(features)
        G = nx.Graph()
        init_k = 5
        clip_feats = features[:, :512].astype(np.float32)
        flow_feats = features[:, 512:].astype(np.float32)
        clip_norms = np.linalg.norm(clip_feats, axis=1, keepdims=True)
        clip_feats = clip_feats / (clip_norms + 1e-6)
        total_blocks = math.ceil(n / Config.graph_block_size)
        with tqdm(total=total_blocks**2, desc="构建动态图") as pbar:
            for i in range(0, n, Config.graph_block_size):
                i_end = min(i + Config.graph_block_size, n)
                block_size_i = i_end - i
                clip_block = clip_feats[i:i_end]
                for j in range(0, n, Config.graph_block_size):
                    j_end = min(j + Config.graph_block_size, n)
                    block_size_j = j_end - j
                    clip_sim_block = np.dot(clip_block, clip_feats[j:j_end].T)
                    flow_block = flow_feats[j:j_end]
                    flow_dist_block = np.sqrt(
                        np.sum((flow_feats[i:i_end, np.newaxis] - flow_block)**2, axis=2))
                    time_diff = np.abs(np.arange(i, i_end)[:, None] - np.arange(j, j_end))
                    time_penalty = 1 + Config.time_decay * time_diff
                    combined_sim = (Config.clip_weight * clip_sim_block + 
                                   (1-Config.clip_weight) * np.exp(-flow_dist_block)) / time_penalty
                    dynamic_k = max(3, init_k - (i//(n//10)))
                    valid_k = min(dynamic_k, block_size_j)
                    for local_i in range(block_size_i):
                        global_i = i + local_i
                        if valid_k <= 0: continue
                        kth = min(valid_k - 1, block_size_j - 1)
                        top_k = np.argpartition(-combined_sim[local_i], kth)[:valid_k]
                        for local_j in top_k:
                            global_j = j + local_j
                            if global_i != global_j and combined_sim[local_i, local_j] > 0:
                                G.add_edge(global_i, global_j, weight=combined_sim[local_i, local_j])
                    pbar.update(1)
        for i in range(n):
            G.add_node(i, feature=features[i])
        return G

    def process(self, frames, fps):
        features = self.extractor.extract_features(frames)
        G = self.build_dynamic_graph(features, fps)
        G = graph_propagation(G)
        return detect_boundaries(G, fps)
