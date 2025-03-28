import os
import csv
import torch
from pathlib import Path
from tqdm import tqdm
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init

NUM_FRAMES = 16
BATCH_SIZE = 4

def init_model():
    disable_torch_init()
    model_path = 'DAMO-NLP-SG/VideoLLaMA2.1-7B-16F'
    
    model, processor, tokenizer = model_init(model_path, device_map="cuda:0")
    
    return model.half().eval(), processor, tokenizer

def process_files(video_files, output_csv):
    model, processor, tokenizer = init_model()
    
    abnormal_prompt = "prompt"
    
    with open(output_csv, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'score', 'status'])
        
        for path_str in tqdm(video_files, desc="Processing"):
            path = Path(path_str)
            if not path.exists():
                writer.writerow([str(path), -1, "Error: File not found"])
                continue
            try:
                video_input = processor['video'](str(path)).to("cuda:0")
                
                with torch.cuda.amp.autocast(), torch.no_grad():
                    output = mm_infer(
                        video_input,
                        abnormal_prompt,
                        model=model,
                        tokenizer=tokenizer,
                        do_sample=False,
                        modal='video'
                    )
                
                try:
                    score = float(output.strip())
                except ValueError:
                    score = -1
                    status = f"Error: Invalid output format - {output}"
                else:
                    status = "success"
                
                writer.writerow([str(path), score, status])
            except Exception as e:
                writer.writerow([str(path), -1, f"Error: {str(e)}"])
            finally:
                torch.cuda.empty_cache()

def main(input_csv, output_csv):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Manifest文件不存在: {input_csv}")
    
    output_dir = os.path.dirname(output_csv)
    os.makedirs(output_dir, exist_ok=True)

    original_entries = []
    video_files = []
    with open(input_csv, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                path = parts[0]
                start = parts[1]
                end = parts[2]
                video_files.append(path)
                original_entries.append({
                    'path': path,
                    'start': start,
                    'end': end,
                    'score': None
                })
    
    process_files(video_files, output_csv)
    
    scores_dict = {}
    with open(output_csv, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            path = row[0]
            score = row[1]
            scores_dict[path] = float(score) if row[2] == "success" else -1

    with open(output_csv, 'w') as f:
        for entry in original_entries:
            line = f"{entry['path']} {entry['start']} {entry['end']} {scores_dict.get(entry['path'], -1)}\n"
            f.write(line)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()
    main(args.input_csv, args.output_csv)



