from pynvml import *
import torch 

def print_gpu_utilization():
    nvmlInit()
    device_count = nvmlDeviceGetCount()
    for i in range(device_count):
        handle = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(handle)
        print(f"GPU {i} memory occupied: {info.used//1024**2} MB, available: {info.free//1024**2} MB.")

def print_available_vram():
    nvmlInit()
    device_count = nvmlDeviceGetCount()
    for i in range(device_count):
        handle = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(handle)
        print(f"GPU {i} available VRAM: {info.free//1024**2} MB.")

def test_load_gpu(dims: tuple = (30000,30000), device: str = "cuda:0"):
    t = torch.ones(dims).to(device)
    print_gpu_utilization()
    # Delete the tensor from GPU to free memory 
    del t 
    torch.cuda.empty_cache()
    print_gpu_utilization()