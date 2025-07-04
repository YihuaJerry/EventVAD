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
