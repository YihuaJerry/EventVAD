import numpy as np
from sklearn.metrics import roc_auc_score
import os

def load_relevant_annotations(annotation_path, target_videos):
    """仅加载目标视频的标注"""
    annotations = {}
    with open(annotation_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            video_name = parts[0]
            if video_name not in target_videos:
                continue
            # 解析异常区间
            intervals = []
            # 第一个异常区间
            s1, e1 = int(parts[2]), int(parts[3])
            if s1 != -1 and e1 != -1:
                intervals.append((s1, e1))
            # 第二个异常区间
            s2, e2 = int(parts[4]), int(parts[5])
            if s2 != -1 and e2 != -1:
                intervals.append((s2, e2))
            annotations[video_name] = intervals
    return annotations

def get_target_videos(model_output_path):
    """从模型输出中提取需要处理的视频列表"""
    target_videos = set()
    with open(model_output_path, 'r') as f:
        for line in f:
            path = line.strip().split()[0]
            video_dir = os.path.basename(os.path.dirname(path))  # 获取目录名
            video_name = video_dir.replace("_segments", "") + ".mp4"
            target_videos.add(video_name)
    return target_videos

def process_model_output(model_output_path, annotations):
    """处理模型输出并生成帧级数据"""
    video_data = {}
    
    # 第一阶段：收集所有视频的时间片段
    with open(model_output_path, 'r') as f:
        for line in f:s
            parts = line.strip().split()
            path = parts[0]
            s = int(parts[1])
            e = int(parts[2])
            score = float(parts[3])
            
            # 提取视频名称（与get_target_videos保持同步）
            video_dir = os.path.basename(os.path.dirname(path))
            video_name = video_dir.replace("_segments", "") + ".mp4"
            
            if video_name not in annotations:
                continue  # 跳过无标注的视频
                
            if video_name not in video_data:
                video_data[video_name] = {'segments': [], 'max_frame': 0}
            
            video_data[video_name]['segments'].append((s, e, score))
            video_data[video_name]['max_frame'] = max(video_data[video_name]['max_frame'], e)
    
    # 第二阶段：生成帧级数据
    all_scores, all_labels = [], []
    for video_name, data in video_data.items():
        # 生成分数数组
        scores = np.zeros(data['max_frame'], dtype=np.float32)
        for s, e, score in sorted(data['segments'], key=lambda x: x[0]):
            end = min(e, data['max_frame'])
            if s < end:
                scores[s:end] = score
        
        # 生成标签数组
        labels = np.zeros(data['max_frame'], dtype=int)
        for s_ann, e_ann in annotations[video_name]:
            e_adj = min(e_ann, data['max_frame'] - 1)
            if s_ann <= e_adj:
                labels[s_ann:e_adj+1] = 1
        
        all_scores.extend(scores.tolist())
        all_labels.extend(labels.tolist())
    
    return all_scores, all_labels

# === 修改 main 函数定义 ===
def main(model_output_path, auc_output_path):  # 添加参数
    # === 删除内部 argparse 代码 ===
    annotation_path = "/tag.txt"
    
    target_videos = get_target_videos(model_output_path)
    annotations = load_relevant_annotations(annotation_path, target_videos)
    scores, labels = process_model_output(model_output_path, annotations)
    
    if len(scores) > 0:
        auc = roc_auc_score(labels, scores)
        with open(auc_output_path, 'w') as f:
            f.write(f"{auc:.4f}")
    else:
        with open(auc_output_path, 'w') as f:
            f.write("0.0")

if __name__ == "__main__":
    # === 保留外部参数解析 ===
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_output", required=True)
    parser.add_argument("--auc_output", required=True)
    args = parser.parse_args()
    main(args.model_output, args.auc_output)  # 正确传递参数
