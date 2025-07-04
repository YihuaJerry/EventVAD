import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
from config import Config

def video_to_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件：{video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    ratio = min(Config.max_resolution[0]/width, Config.max_resolution[1]/height)
    new_size = (int(width*ratio), int(height*ratio)) if ratio < 1 else (width, height)
    new_size = (new_size[0]//2*2, new_size[1]//2*2)
    
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
