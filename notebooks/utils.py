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

def batch_inference(encoding, model, tokenizer, terminators, max_tokens: int = 20):
  import timeit

  start_time = timeit.default_timer()

  with torch.no_grad():
      generated_ids = model.generate(
          **encoding,
          pad_token_id = tokenizer.pad_token_id,
          max_new_tokens=max_tokens,
          eos_token_id=terminators,
          do_sample=True,
          temperature=0.6,
          top_p=0.9,
      )

  batch_generation_elapsed = timeit.default_timer() - start_time
  print(f"Time taken: {batch_generation_elapsed} seconds")

  return generated_ids
