import torch
import time
import sys

def hold_gpus():
    """
    Allocates a small tensor on all visible CUDA devices to reserve them.
    This process then sleeps indefinitely until it is killed.
    """
    if not torch.cuda.is_available():
        print("INFO: CUDA not available, placeholder script exiting.", file=sys.stderr)
        return

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("INFO: No visible CUDA devices, placeholder script exiting.", file=sys.stderr)
        return

    print(f"INFO: GPU placeholder reserving {num_gpus} GPU(s)...", file=sys.stderr)

    placeholders = []
    for i in range(num_gpus):
        try:
            # Allocate ~256MB on each GPU. This is usually enough to prevent
            # other users' schedulers from seeing the GPU as "free".
            tensor = torch.ones((1, 1024, 1024, 64), dtype=torch.float32, device=f'cuda:{i}')
            placeholders.append(tensor)
            print(f"INFO:  - Reserved GPU cuda:{i} ({torch.cuda.get_device_name(i)})", file=sys.stderr)
        except Exception as e:
            print(f"WARNING: Could not reserve memory on GPU cuda:{i}. Error: {e}", file=sys.stderr)

    print("INFO: GPU(s) reserved. Placeholder is now idle.", file=sys.stderr)
    try:
        # Sleep indefinitely until killed
        while True:
            time.sleep(3600)
    except (KeyboardInterrupt, SystemExit):
        print("\nINFO: GPU placeholder shutting down, releasing GPUs.", file=sys.stderr)

if __name__ == "__main__":
    hold_gpus()