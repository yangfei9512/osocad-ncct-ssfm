import os
import shutil
import torch
import time
import datetime
import subprocess
import socket
import sys
from pathlib import Path

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TORCHRUN = os.path.join(os.path.dirname(sys.executable), "torchrun")
DATASET_ROOT = Path(os.environ.get("INTERNAL_DATASET_ROOT", "data/internal"))
DATA_ROOT = DATASET_ROOT / "images"
TRAIN_JSON = DATASET_ROOT / "json" / "train.json"
VAL_JSON = DATASET_ROOT / "json" / "validation.json"

# Current timestamp for uniquely identifying the experiment
TIME_NOW = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
BASE_CHECKPOINT_PATH = os.path.join(THIS_DIR, "runs", f"{TIME_NOW}_launch")

def scale_lr(base_lr, bs, base_bs=64):
    scaled_lr = round(base_lr * (bs / base_bs), 8)
    return scaled_lr

def copy_all_py_files(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for fname in os.listdir(src_dir):
        if fname.endswith(".py"):
            shutil.copy2(os.path.join(src_dir, fname), os.path.join(dst_dir, fname))

def find_free_port():
    """Find a random free port."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port

# Parameter grid
base_lr_list = [1e-5,2e-5]
bs = 16
drop_out = 0.2
weight_decay = 0.1
num_layers_list = [4,6,8,16]
image_size = 224
max_slices = 32
target_slices = 32
num_epochs = 1000
num_workers = 4
loss_name = "focal"
focal_gamma = 2.0

# GPU control
num_gpus = torch.cuda.device_count()
assert num_gpus >= 1, f"At least 1 GPU is required, found {num_gpus}"

use_num_gpus = 1  # Number of GPUs used per experiment
assert use_num_gpus <= num_gpus, (
    f"Each experiment requests {use_num_gpus} GPUs, but only {num_gpus} are available"
)
max_parallel_runs = num_gpus // use_num_gpus
processes = []
run_id = 0

# Create experiment directories and copy code files
os.makedirs(BASE_CHECKPOINT_PATH, exist_ok=True)
copy_all_py_files(src_dir=THIS_DIR, dst_dir=BASE_CHECKPOINT_PATH)

for base_lr in base_lr_list:
    for num_layers in num_layers_list:
        lr = scale_lr(base_lr, bs, bs)
        checkpoint_path = os.path.join(BASE_CHECKPOINT_PATH, f"run_{run_id}")
        os.makedirs(checkpoint_path, exist_ok=True)

        # Assign GPU
        start_gpu = (run_id * use_num_gpus) % num_gpus
        cuda_devices = ",".join(str(i % num_gpus) for i in range(start_gpu, start_gpu + use_num_gpus))

        # Random free port to reduce conflicts
        master_port = find_free_port()

        cmd = (
            f"CUDA_VISIBLE_DEVICES={cuda_devices} {TORCHRUN} "
            f"--nproc_per_node={use_num_gpus} --master_port={master_port} "
            f"{os.path.join(THIS_DIR, 'train.py')} "
            f"--CHECKPOINT_PATH {checkpoint_path} "
            f"--data_root {DATA_ROOT} "
            f"--train_json {TRAIN_JSON} "
            f"--val_json {VAL_JSON} "
            f"--batch_size {bs} "
            f"--lr {lr} "
            f"--drop_out {drop_out} "
            f"--weight_decay {weight_decay} "
            f"--image_size {image_size} "
            f"--max_slices {max_slices} "
            f"--target_slices {target_slices} "
            f"--num_layers {num_layers} "
            f"--num_epochs {num_epochs} "
            f"--num_workers {num_workers} "
            f"--loss {loss_name} "
            f"--focal_gamma {focal_gamma}"
        )

        print(f"[Run {run_id}] GPUs: {cuda_devices}, Port: {master_port}")
        print(f"Command: {cmd}")

        # Logging
        log_file = os.path.join(checkpoint_path, "log.txt")
        with open(log_file, "w") as f:
            f.write(f"[Start time] {TIME_NOW}\n")
            f.write(f"[Command] {cmd}\n\n")

        log_f = open(log_file, "a")
        process = subprocess.Popen(cmd, shell=True, stdout=log_f, stderr=log_f)
        processes.append((process, log_f))
        time.sleep(2)
        run_id += 1

        if len(processes) >= max_parallel_runs:
            process, process_log = processes.pop(0)
            process.wait()
            process_log.close()

# Wait for any remaining processes to finish.
for p, log_f in processes:
    p.wait()
    log_f.close()
