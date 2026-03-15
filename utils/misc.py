import random
import numpy as np
import torch

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def length2mask(length, size=None):
    batch_size = len(length)
    size = int(max(length)) if size is None else size
    device = length.device if isinstance(length, torch.Tensor) else torch.device('cpu')
    length_tensor = length if isinstance(length, torch.Tensor) else torch.tensor(length, dtype=torch.int64, device=device)
    mask = (
        torch.arange(size, dtype=torch.int64, device=device).unsqueeze(0).repeat(batch_size, 1)
        > (length_tensor.long() - 1).unsqueeze(1)
    )
    return mask
