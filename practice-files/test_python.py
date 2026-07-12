import torch
import time


print(f"Gpu is available: {torch.cuda.is_available()}")
print(f"Cuda version is : {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory of gpu is : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# checking memory


size = 5000

# making tensors
a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

start = time.time()  # starting the timer
#print(start)

c_cpu = a_cpu @ b_cpu

cpu_time = time.time() - start
print(f"time taken by cpu {cpu_time:.2f}s")
# gpu 

if torch.cuda.is_available():
    a_gpu = a_cpu.to("cuda")
    b_gpu = b_cpu.to("cuda")
    torch.cuda.synchronize()
    start = time.time()
    c_gpu = a_gpu @ b_gpu

    torch.cuda.synchronize()
    
    gpu_time = time.time() - start

    print(f"time taken by gpu: {gpu_time:.2f}s")
    print(f"the difference of time was : {cpu_time / gpu_time:.1f}s")