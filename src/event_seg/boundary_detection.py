import numpy as np
from scipy.signal import savgol_filter
from config import Config

def detect_boundaries(G, fps):
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
