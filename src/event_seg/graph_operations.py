import os
import cv2
import glob
from config import Config
from utils import video_to_frames
from uniseg_processor import UniSegProcessor

def process_video(input_path, output_dir):
    processor = UniSegProcessor()

    # 获取视频基本信息
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        cap = cv2.VideoCapture(input_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件：{input_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # 计算新尺寸
    ratio = min(Config.max_resolution[0]/original_width, 
               Config.max_resolution[1]/original_height)
    new_size = (int(original_width*ratio), int(original_height*ratio)) if ratio < 1 else (original_width, original_height)
    new_size = (new_size[0]//2*2, new_size[1]//2*2)  # 确保尺寸为偶数
    
    # 读取帧
    frames, fps, new_size = video_to_frames(input_path) 
    
    # 处理视频获取边界
    boundaries = processor.process(frames, fps)

    # 转换边界为帧索引
    frame_boundaries = []
    if boundaries:
        for start_sec, end_sec in boundaries:
            start_frame = int(round(start_sec * fps))
            end_frame = min(int(round(end_sec * fps)), total_frames - 1)
            if end_frame - start_frame > 1:
                frame_boundaries.append((start_frame, end_frame))
        
        # 添加起始和结束边界
        if frame_boundaries and frame_boundaries[0][0] > 0:
            frame_boundaries.insert(0, (0, frame_boundaries[0][0]))
        
        last_end = frame_boundaries[-1][1] if frame_boundaries else 0
        if last_end < total_frames - 1:
            frame_boundaries.append((last_end, total_frames - 1))
    else:
        frame_boundaries.append((0, total_frames - 1))

    # 保存分段视频
    if frame_boundaries:
        codec_candidates = [
            ('mp4v', 'mp4v'),
            ('avc1', 'avc1'),
            ('xvid', 'XVID')
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
