import torch


def cuda_setup():
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        torch.cuda.empty_cache()
    
    n_gpu = torch.cuda.device_count()
    device = torch.device("cuda" if is_cuda else "cpu")
    return n_gpu, device