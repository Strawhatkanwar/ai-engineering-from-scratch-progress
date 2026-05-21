## This file will build tools which will help us in debugging our ai workflows in terms of 
## small error that persists like shape mismatches, tensors types, logging, bassis OOM's etc 
## also here i solve the exercise questions of this ai-engineering-from-scratch course debug tools.


import logging
import time
import sys
import tracemalloc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# part 3: python logging
# we can replace print statements with logging when our debugging goes beyond a quick check.
# logging gives us time stamps, severtiy levels, and file output. When a training run fails let's say
# at 4AM, we want a log file, no terminal output that scrolled off screen.

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from torch.utils.tensorboard import SummaryWriter



# -------------------------------starting with pyton print---------------------------------
def debug_print(name, tensor):
    '''
    this is targeted print statement for code involving the tensors, it will print
    shapes, dtypes, value range, and devices
    '''
    print(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, "
          f"device={tensor.device}, "
          f"min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, "
          f"mean={tensor.mean().item():.4f}, "
          f"has_nan={tensor.isnan().any().item()}")

# part 2 : python debugger (pbd and breakpoint)

def training_step(model, batch, criterion, optimizer):
    """
    putting breakpoint() in our training loop and inspect tensors interatively
    """
    inputs, labels = batch
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    
    if loss.item() > 100 or torch.isnan(loss):
        breakpoint()
    
    
    loss.backward()
    optimizer.step()




# logger.info("Starting training: lr=%.4f, batch_size=%d", lr, batch_size)
# logger.warning("Loss spike detected: %.4f at step %d", loss.item(), step)
# logger.error("NaN loss at step %d, stopping", step)


# part 4: Timing code Sections
# common findings include the data loading itself is takeing like 60% of our time. 
# the fix for that is `num_workers > 0` in our Data Loader , not a faster gpu.


class Timer:
    def __init__(self, name=""):
        self.name = name
        
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start
        print(f"[{self.name}] {elapsed:.4f}s")


# @profile
# def load_data():
#     raw = read_csv("data.csv")  # watch memory jump here
#     processed = preprocess(raw)  # and here
#     return processed

# run with `python -m memory_profiler my_script.py` to see line-by-line memory usage.


def check_gpu():
    if torch.cuda.is_available():
        print(torch.cuda.memory_summary())

        print(f"Allocated : {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# part 7: Common ai bugs and how to catch them.
# most common bug is "a tensor has shape [batch, features] when the model expects [batch, channels, height, width]"

def check_shapes(model, sample_input):
    # run this funntion with a sample batch. It maps every shape transformtation in our model.

    print(f"Input: {sample_input.shape}")
    hooks = []

    def make_hook(name):
        def hook(module, inp, out):
            in_shape = inp[0].shape if isinstance(inp, tuple) else inp.shape
            out_shape = out.shape if hasattr(out, "shape") else type(out)
            print(f" {name}: {in_shape} --> {out_shape}")
        return hook
    
    for name, module in model.named_modules():
        hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        model(sample_input)

    for h in hooks:
        h.remove()


#### Nan Loss
'''
Nan loss means something exploded. Common causes:
- learning rate too high
- Division by zero in custom loss
- Log of zero or negative number
- Exploding gradients in RNNs
'''

def detect_nan(model, loss, step):
    if torch.isnan(loss):
        print(f"NAN loss at each step {step}")
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f" NaN gradient in {name}")
                if torch.isinf(param.grad).any():
                    print(f" Inf gradient in {name}")

        return True
    return False


#### Data Leakage
# our model gets 99% accuracy in test set. sounds great, it's a bug
# Also check for temporal leakage: using future data to predict the past.
# Sort by timestamp before splitting.

def check_data_leakage(train_set, test_set, id_column="id"):
    train_ids = set(train_set[id_column].tolist())
    test_ids = set(test_set[id_column].tolist())
    overlap = train_ids & test_ids
    if overlap:
        print(f"DATA LEAKAGE; {len(overlap)} samples in both train and test")
        return True
    return False

### Wrong device

# tensors on different devices (CPU vs GPU) causes runtime errors. But sometimes a tensor
# silently stays on CPU while everything else is on GPU, and training just runs slowly.

def check_devices(model, *tensors):
    model_device = next(model.parameters()).device
    print(f"Model device: {model_device}")
    for i, t in enumerate(tensors):
        if t.device != model_device:
            print(f" WARNING: tensor {i} on {t.device}, model on {model_device}")


def check_gradient_health(model):
    total_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            total_norm += grad_norm ** 2
            if grad_norm > 100:
                print(f"    WARNING: large gradient in {name}: {grad_norm:.2f}")
            if grad_norm == 0:
                print(f"    WARNING: zero gradient in {name}")
    total_norm = total_norm ** 0.5
    print(f"  Total gradient norm: {total_norm:.4f}")
    return total_norm

### tensorboard basics(exericse 4)


# simple model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# fake train and val data
X_train = torch.randn(1000, 784)
y_train = torch.randint(0, 10, (1000, ))
X_val = torch.randn(200, 784)
y_val = torch.randint(0, 10, (200, ))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

writer = SummaryWriter("runs/experiment_1")
num_steps = 100
for step in range(num_steps):
    
    model.train()  # training
    optimizer.zero_grad()
    output = model(X_train)  # forward pass
    
    # breakpoint()   # exercise 5
    
    train_loss = criterion(output, y_train)
    train_loss.backward()
    optimizer.step()

    # validation
    model.eval()
    with torch.no_grad():
        val_output = model(X_val)
        val_loss = criterion(val_output, y_val)
    
    logger.info("step %d: train=%.4f val=%.4f", step, train_loss, val_loss)

    if train_loss > val_loss:
        logger.warning("Model might be under fitting at epoch %d", step)

    # log both losses 
    writer.add_scalar("loss/train", train_loss.item(), step)
    writer.add_scalar("loss/val", val_loss.item(), step)

    if step % 10 == 0:
        print(f"epoch {step}: train={train_loss.item():.4f} val={val_loss.item():.4f}")
        for name, param in model.named_parameters():
            writer.add_histogram(f"weights/{name}", param, step)
            if param.grad is not None:
                writer.add_histogram(f"grads/{name}", param.grad, step)

writer.close()
print("Done! Run! tensorboard")

## ------------------finish exercise 4-----------------------------------

# launch it
# tensorboard --logdir=runs
# look for loss not decreasing: learning rate too low, or model architecture issue
# train loss decreasing, val loss increasing: Overfitting
# weight histograms collapsing to zero: Vanishing gradietns
# gradient histograms exploding: Need gradient clipping



# writing --------------------demos for above funtoins to try--------------------------

def demo_print_debugging():
    print("\n---1. Print  Debugging for tensors ----")
    x = torch.randn(32, 784)
    debug_print("input_batch", x)

    w = torch.randn(784, 128)
    out = x @ w
    debug_print("after_matmul", out)

    with_nan = out.clone()
    with_nan[0, 0] = float("nan")
    debug_print("with_added_nan", with_nan)

@profile  # adding profile
def demo_timing():
    print("\n----2. timing code sections ")

    with Timer("matrix multiply 1000x1000"):
        a = torch.randn(1000, 1000)
        b = torch.randn(1000, 1000)
        _ = a @ b

    with Timer("matrix multiply 5000x5000"):
        a = torch.randn(5000, 5000)
        b = torch.randn(5000, 5000)
        _ = a @ b

def demo_memory_tracking():
    print("\n ----3. Memory tracking(tracemalloc)...")
    tracemalloc.start()

    data = [torch.randn(100, 100) for _ in range(100)]
    more_data = torch.randn(1000, 1000)

    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics("lineno")
    
    print("  Top 5 memory allocations:")
    for stat in top_stats[:5]:
        print(f"    {stat}")

    del data, more_data
    tracemalloc.stop()


def demo_shape_checking():
    print("\n---4. Checking shapes throug model...")

    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )
    sample = torch.randn(4, 784)
    check_shapes(model, sample)

def demo_nan_detection():
    print("\n ----5: NAN detection....")

    model = nn.Sequential(
        nn.Linear(784, 265),
        nn.ReLU(),
        nn.Linear(265, 10),

    )
    x = torch.randn(4, 784)
    target = torch.randint(0, 10, (4,))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    optimizer.zero_grad()
    output = model(x)
    output = output/ 0
    loss = criterion(output, target)
    loss.backward()
    print(f"Normal loss... {loss.item():.4f}")
    nan_found = detect_nan(model, loss, step=0)
    print(f"nan detected.....{nan_found}")

    fake_nan_loss = torch.tensor(float("nan"))
    print(f" Simulated NaN loss: {fake_nan_loss.item()}")
    nan_found = detect_nan(model, fake_nan_loss, step=99)
    print(f"Nan detected ...{nan_found} ")

def demo_device_checking():
    print("\n --- 6. Device checking.............")
    model = nn.Linear(10, 5)
    t1 = torch.randn(4, 10)
    t2 = torch.randn(4, 10)

    check_devices(model, t1, t2)

    if torch.cuda.is_available():
        model_gpu = model.cuda()
        t_cpu = torch.randn(4, 10)
        t_gpu = torch.randn(4, 10).cuda()
        print("with mixed devices....")
        check_devices(model_gpu, t_cpu, t_gpu)

def demo_gradient_health():
    print("\n...7. gradient health......")
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    x = torch.randn(4, 784)
    target = torch.randint(0, 10, (4,))
    criterion = nn.CrossEntropyLoss()

    output = model(x)
    loss = criterion(output, target)
    loss.backward()
    check_gradient_health(model)


def demo_gpu_memory():
    print("\n ---8. gpu memory summary")
    if not torch.cuda.is_available():
        print("  No GPU available. Skipping GPU memory demo.")
        print("  On a GPU machine, torch.cuda.memory_summary() shows:")
        print("    - Allocated memory per block size")
        print("    - Cached (reserved) memory")
        print("    - Peak memory usage")
        return
    
    print(f" GPU: {torch.cuda.get_device_name(0)}")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e6:.1f}MB")
    print(f" Cached: {torch.cuda.memory_reserved() / 1e6:.1f}MB")

    large_tensor = torch.randn(10000, 10000, device="cuda")
    print(f" After 10k x 10k tensor: ")
    print(f" Allocated gpu memory: {torch.cuda.memory_allocated() / 1e6:.1f}MB")

    del large_tensor
    torch.cuda.empty_cache()
    print(f" after delteing the tensor from memory: ")
    print(f"After clean the allocated memory : {torch.cuda.memory_allocated() / 1e6:.1f}MB ")

def demo_logging():
    print("\n....9. demo of logging")

    logger.info("Training started: lr=0.001, batch_size=32, epochs=10")
    logger.info("step 100: loss=2.306, accuracy=0.10")
    logger.warning("Loss spike detected: 15.7 at step 450")
    logger.info("step 1000: loss=0.4514, Accuracy=0.87")
    logger.info("Training complete: best_loss=0.3201")

def demo_conditional_breakpoint():
    print("\n--- 10. Conditional Breakpoint Pattern ---")
    print("  In real code, use this pattern:")
    print()
    print("    for step in range(num_steps):")
    print("        loss = train_step(model, batch)")
    print("        if loss.item() > 10 or torch.isnan(loss):")
    print("            breakpoint()  # drops into pdb")
    print()
    print("  Useful pdb commands once inside:")
    print("    p tensor.shape       # print shape")
    print("    p tensor.device      # check device")
    print("    p tensor.grad        # inspect gradients")
    print("    p tensor.isnan().sum()  # count NaNs")
    print("    c                    # continue execution")
    print("    q                    # quit debugger")

# -----------------exercise 3---------------------------------

def demo_tracemalloc():
    print("\n ----3b. Tracemalloc on data pipeline: ....")
    tracemalloc.start()

    #simulating data
    raw_data = [i for i in range(100000)]        # line A - loading raw data
    processed = [x * 2 for x in raw_data]        # line B - preprocessing
    batches = [processed[i: i+32] for i in range(0, len(processed), 32)] # line c - batching
    tensor_batches = [torch.tensor(b) for b in batches]  # to tensors

    snapshot = tracemalloc.take_snapshot()
    stats = snapshot.statistics("lineno")

    print("Top five memory allcoations")
    for stat in stats[:5]:
        print(f"  {stat}")
    
    tracemalloc.stop()

def main():
    print("=" * 60)
    print("  AI Debugging and Profiling Toolkit")
    print("  Phase 0, Lesson 12")
    print("=" * 60)

    if not HAS_TORCH:
        print("\nPyTorch not installed. Install with:")
        print("  uv pip install torch")
        print("\nRunning non-PyTorch demos only...\n")
        demo_memory_tracking()
        demo_logging()
        return 1
    
    demo_print_debugging()
    demo_timing()
    demo_memory_tracking()
    demo_shape_checking()
    demo_nan_detection()
    demo_device_checking()
    demo_gradient_health()
    demo_gpu_memory()
    demo_logging()
    demo_conditional_breakpoint()
    demo_tracemalloc()

    print("\n" + "=" * 60)
    print("  All demos complete.")
    print("  Next: introduce bugs intentionally and practice catching them.")
    print("=" * 60 + "\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())