import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import cv2
import torch
import numpy as np
from tqdm import tqdm
import networkx as nx
from lavis.models import load_model_and_preprocess
import sys
import pathlib
project_root = pathlib.Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))
from RAFT.core.raft import RAFT
from scipy.signal import savgol_filter
from argparse import Namespace
from PIL import Image
from collections import defaultdict
import math
import glob


# ------------------ 配置 ------------------
class Config:
    device = "cuda"
    fp16_enabled = False
    
    # 特征提取
    clip_model_name = "clip"
    feature_dim = 640
    
    # 动态图参数
    time_decay = 0.05
    clip_weight = 0.8
    
    # 边界检测参数
    ema_window = 2.0
    
    # 系统
    max_resolution = (1280, 720)
    chunk_size = 500
    graph_block_size = 200
    ortho_dim = 64
    gat_iters = 1
    mad_multiplier = 3.0
    min_segment_gap = 2.0
    raft_iters = 20

# ------------------ 读取帧 ------------------
def video_to_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        # 尝试用FFmpeg重新打开
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件：{video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    ratio = min(Config.max_resolution[0]/width, Config.max_resolution[1]/height)
    new_size = (int(width*ratio), int(height*ratio)) if ratio < 1 else (width, height)
    new_size = (new_size[0]//2*2, new_size[1]//2*2)  # 确保尺寸为偶数
    
    frames = []
    for idx in tqdm(range(0, total_frames), desc="读取帧"):
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), new_size)
            frames.append(frame.astype(np.uint8))
        
        if len(frames) % 100 == 0:
            torch.cuda.empty_cache()
    
    cap.release()
    return np.array(frames), fps, new_size 

class FeatureExtractor:  
    def __init__(self):  
        model_info = load_model_and_preprocess(
            name=Config.clip_model_name,
            model_type="ViT-B-16",
            is_eval=True,
            device=Config.device
        )
        self.clip_model = model_info[0].float()
        self.vis_processors = model_info[1]
        raft_args = Namespace(
            model='/Node09_nvme/hhj/gnn_method/RAFT/models/raft-things.pth',
            small=False,
            mixed_precision=False,
            alternate_corr=False,
            dropout=0.0
        )
        self.raft_model = RAFT(raft_args).to(Config.device).eval().float()
        self.flow_proj = self._init_random_ortho(2, 128)

    def _init_random_ortho(self, in_dim, out_dim):
        np.random.seed(42)
        q, _ = np.linalg.qr(np.random.randn(max(in_dim, out_dim), max(in_dim, out_dim)))
        return q[:in_dim, :out_dim].astype(np.float32)

    def extract_clip_features(self, frames):
        features = []
        for i in range(0, len(frames), Config.chunk_size):
            chunk = frames[i:i+Config.chunk_size]
            with torch.no_grad():
                processed = torch.stack([
                    self.vis_processors["eval"](Image.fromarray(f)) 
                    for f in chunk
                ]).to(Config.device).float()                
                chunk_feats = self.clip_model.encode_image(processed)
                features.append(chunk_feats.cpu().numpy())
            del processed, chunk_feats
            torch.cuda.empty_cache()
        return np.concatenate(features)

    def extract_flow_features(self, frames):
        if len(frames) < 2:
            return np.zeros((len(frames), 128))
        flow_features = []
        prev_frame = torch.tensor(frames[0], device=Config.device).permute(2,0,1).unsqueeze(0).float()
        for i in tqdm(range(1, len(frames)), desc="光流特征提取"):
            curr_frame = torch.tensor(frames[i], device=Config.device).permute(2,0,1).unsqueeze(0).float()
            with torch.no_grad():
                flow = self.raft_model(prev_frame, curr_frame, iters=Config.raft_iters)[-1]
                pooled = torch.mean(flow, dim=[2,3]).cpu().numpy().flatten()
            flow_features.append(pooled)
            prev_frame = curr_frame.clone()
            del curr_frame, flow
            if i % 100 == 0:
                torch.cuda.empty_cache()
        flow_features.insert(0, np.zeros_like(flow_features[0]))
        projected = np.dot(np.array(flow_features), self.flow_proj)
        return projected.reshape(len(frames), -1)

    def extract_features(self, frames):
        clip_feats = self.extract_clip_features(frames)
        flow_feats = self.extract_flow_features(frames)
        assert clip_feats.shape[0] == flow_feats.shape[0]
        return np.concatenate([
            Config.clip_weight * clip_feats,
            (1-Config.clip_weight) * flow_feats
        ], axis=1).astype(np.float32)

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

    def graph_propagation(self, G):
        node_list = list(G.nodes)
        features = np.array([G.nodes[i]['feature'] for i in node_list], dtype=np.float32)
        np.random.seed(42)
        Q = np.linalg.qr(np.random.randn(Config.feature_dim, Config.ortho_dim))[0]
        K = np.linalg.qr(np.random.randn(Config.feature_dim, Config.ortho_dim))[0]
        V = np.linalg.qr(np.random.randn(Config.feature_dim, Config.feature_dim))[0]
        block_size = Config.chunk_size
        for _ in range(Config.gat_iters):
            attn = np.zeros((len(node_list), len(node_list)), dtype=np.float32)
            for i in range(0, len(node_list), block_size):
                i_end = i + block_size
                chunk_Q = features[i:i_end] @ Q
                for j in range(0, len(node_list), block_size):
                    j_end = j + block_size
                    chunk_K = features[j:j_end] @ K
                    attn[i:i_end, j:j_end] = chunk_Q @ chunk_K.T
            adj = nx.adjacency_matrix(G).toarray()
            attn = np.where(adj > 0, attn, -np.inf)
            attn = torch.softmax(torch.tensor(attn).float(), dim=1).numpy()
            messages = np.zeros_like(features)
            for i in range(0, len(node_list), block_size):
                i_end = i + block_size
                messages[i:i_end] = attn[i:i_end] @ (features @ V)
            features += 0.5 * messages
            features -= features.mean(0)
        for i, node in enumerate(node_list):
            G.nodes[node]['feature'] = features[i]
        return G

    def detect_boundaries(self, G, fps):
        features = np.array([G.nodes[i]['feature'] for i in G.nodes])
        diffs = np.diff(features, axis=0)
        s = np.linalg.norm(diffs, axis=1)**2
        cos_sim = np.array([
            np.dot(features[i], features[i+1]) / 
            (np.linalg.norm(features[i]) * np.linalg.norm(features[i+1]) + 1e-6)
            for i in range(len(features)-1)
        ])
        s_cos = 1 - cos_sim
        s_combined = s + s_cos
        window_size = max(1, int(fps * Config.ema_window))
        if len(s_combined) < window_size*2: return []
        s_smoothed = savgol_filter(s_combined, window_length=window_size, polyorder=2)
        ema = np.convolve(s_smoothed, np.ones(window_size)/window_size, mode='valid')
        s_ratio = s_smoothed[window_size-1:] / (ema + 1e-6)
        median = np.median(s_ratio)
        mad = np.median(np.abs(s_ratio - median))
        threshold = median + Config.mad_multiplier * mad
        boundaries = np.where(s_ratio > threshold)[0] + window_size//2
        merged = []
        prev = boundaries[0] if boundaries.size > 0 else None
        for b in boundaries[1:]:
            if prev is not None and (b - prev) < Config.min_segment_gap * fps:
                prev = b
            else:
                if prev is not None: merged.append(prev)
                prev = b
        if prev is not None: merged.append(prev)
        time_boundaries = []
        for i in range(len(merged)):
            start = merged[i]/fps
            end = (merged[i+1]/fps if i+1 < len(merged) 
                  else (merged[i] + Config.min_segment_gap * fps)/fps)
            time_boundaries.append( (max(0, start), min(end, len(features)/fps)) )
        return time_boundaries

# ------------------ 视频处理函数修改版 ------------------
def process_video(input_path, output_dir):
    processor = UniSegProcessor()

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    ratio = min(Config.max_resolution[0]/original_width, 
               Config.max_resolution[1]/original_height)
    new_size = (int(original_width*ratio), int(original_height*ratio)) if ratio < 1 else (original_width, original_height)
    new_size = (new_size[0]//2*2, new_size[1]//2*2)  # 确保偶数尺寸
    
    frames, fps, new_size = video_to_frames(input_path) 
    
    features = processor.extractor.extract_features(frames)
    G = processor.build_dynamic_graph(features, fps)
    G = processor.graph_propagation(G)
    boundaries = processor.detect_boundaries(G, fps)

    frame_boundaries = []
    
    if boundaries:
        for start_sec, end_sec in boundaries:
            start_frame = int(round(start_sec * fps))
            end_frame = min(int(round(end_sec * fps)), total_frames - 1)
            if end_frame - start_frame > 1:
                frame_boundaries.append((start_frame, end_frame))
        
        if frame_boundaries and frame_boundaries[0][0] > 0:
            frame_boundaries.insert(0, (0, frame_boundaries[0][0]))
        
        last_end = frame_boundaries[-1][1] if frame_boundaries else 0
        if last_end < total_frames - 1:
            frame_boundaries.append((last_end, total_frames - 1))
    else:
        frame_boundaries.append((0, total_frames - 1))

    if frame_boundaries:
        codec_candidates = [
            ('mp4v', 'mp4v'),  # 原始可用编码器
        ]
        
        selected_codec = None
        for codec_name, fourcc_code in codec_candidates:
            test_path = os.path.join(output_dir, f"test_{codec_name}.mp4")
            test_writer = cv2.VideoWriter(
                test_path, 
                cv2.VideoWriter_fourcc(*fourcc_code),
                fps, 
                new_size
            )
            if test_writer.isOpened():
                test_writer.release()
                os.remove(test_path)
                selected_codec = fourcc_code
                print(f"选择编码器: {fourcc_code}")
                break
        
        if selected_codec is None:
            print("无法找到新编码器，回退mp4v")
            selected_codec = 'mp4v'
        
        cap = cv2.VideoCapture(input_path)
        for seg_idx, (start, end) in enumerate(frame_boundaries):
            output_path = os.path.join(output_dir, f"segment_{seg_idx:04d}.mp4")
            
            out = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*selected_codec),
                fps,
                new_size
            )
            
            if not out.isOpened():
                print(f"视频写入失败，改用PNG序列保存片段 {seg_idx}")
                seg_dir = os.path.join(output_dir, f"segment_{seg_idx:04d}")
                os.makedirs(seg_dir, exist_ok=True)
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                for idx in range(start, end+1):
                    ret, frame = cap.read()
                    if ret:
                        cv2.imwrite(os.path.join(seg_dir, f"frame_{idx:06d}.png"), frame)
                continue
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            current_frame = start
            while current_frame <= end:
                ret, frame = cap.read()
                if ret:
                    out.write(frame)
                    current_frame += 1
                else:
                    break
            out.release()
        
        cap.release()
    
    return frame_boundaries

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="输入视频目录路径")
    parser.add_argument("--output", required=True, help="输出根目录路径")
    args = parser.parse_args()
    manifest_path = os.path.join(args.output, "segment_manifest.txt")
    empty_log_path = os.path.join(args.output, "empty_videos.log")
    os.makedirs(args.output, exist_ok=True)
    if not os.path.exists(manifest_path):
        with open(manifest_path, "w") as f:
            f.write("")
    if not os.path.exists(empty_log_path):
        with open(empty_log_path, "w") as ef:
            ef.write("empty_video_path\n")
    existing_seg_paths = set()
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            lines = f.readlines()
            if lines and lines[0].strip() == "file_path start_frame end_frame":
                for line in lines[1:]:
                    parts = line.strip().split()
                    if parts: existing_seg_paths.add(parts[0])
    for root, dirs, files in os.walk(args.input):
        for file in files:
            if file.split('.')[-1].lower() in ['mp4', 'avi', 'mov']:
                input_path = os.path.join(root, file)
                rel_path = os.path.relpath(root, args.input)
                video_name = os.path.splitext(file)[0]
                output_dir = os.path.join(args.output, rel_path, f"{video_name}_segments")
                processed = False
                if os.path.isdir(output_dir):
                    segments = glob.glob(os.path.join(output_dir, "segment_*.mp4"))
                    if len(segments) > 0: processed = True
                seg_prefix = os.path.join(output_dir, "segment_")
                if any(p.startswith(seg_prefix) for p in existing_seg_paths):
                    processed = True
                if processed:
                    print(f"已跳过: {input_path}")
                    continue
                try:
                    os.makedirs(output_dir, exist_ok=True)
                    segment_info = process_video(input_path, output_dir)
                    if not segment_info:
                        with open(empty_log_path, "a") as ef:
                            ef.write(f"{input_path}\n")
                        print(f"无分割结果: {input_path}")
                        continue
                    with open(manifest_path, "a") as f:
                        for seg_idx, (start_frame, end_frame) in enumerate(segment_info):
                            seg_path = os.path.abspath(os.path.join(output_dir, f"segment_{seg_idx:04d}.mp4"))
                            f.write(f"{seg_path} {start_frame} {end_frame}\n")
                except Exception as e:
                    print(f"处理失败: {input_path} - {str(e)}")
                    continue
    print(f"处理完成！清单文件: {manifest_path}")
    print(f"空视频日志: {empty_log_path}")
