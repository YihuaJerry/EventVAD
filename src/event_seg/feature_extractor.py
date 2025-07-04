import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from argparse import Namespace
from config import Config
from utils import video_to_frames
from lavis.models import load_model_and_preprocess
from RAFT.core.raft import RAFT

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
            model='/path/raft-things.pth',
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
