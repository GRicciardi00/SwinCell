import torch

def is_cuda_available():
    return torch.cuda.is_available()

def is_mps_available():
    try:
        return getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available() and torch.backends.mps.is_built()
    except Exception:
        return False

def get_device(preferred=None):
    """
    Return a torch.device chosen from:
      - preferred if provided and available (e.g. 'cuda', 'mps', 'cpu' or 'cuda:0', 'mps')
      - cuda if available
      - mps if available
      - otherwise cpu
    preferred may be a string or torch.device.
    """
    if preferred is not None:
        if isinstance(preferred, torch.device):
            pstr = preferred.type
        else:
            pstr = str(preferred)
        if pstr.startswith("cuda") and is_cuda_available():
            return torch.device(pstr)
        if pstr.startswith("mps") and is_mps_available():
            return torch.device("mps")
        if pstr == "cpu":
            return torch.device("cpu")

    # prefer CUDA first, then MPS, then CPU
    if is_cuda_available():
        return torch.device("cuda")
    if is_mps_available():
        return torch.device("mps")
    return torch.device("cpu")

def select_distributed_backend():
    """
    Return recommended distributed backend:
      - 'nccl' for CUDA
      - 'gloo' for CPU or MPS
    """
    if is_cuda_available():
        return "nccl"
    return "gloo"