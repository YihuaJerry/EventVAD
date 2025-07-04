import os
import sys
import pathlib
import argparse
import glob
from video_processing import process_video

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    project_root = pathlib.Path(__file__).parent.resolve()
    sys.path.insert(0, str(project_root))
    
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
