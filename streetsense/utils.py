import numpy as np
import torch

def set_seed(seed: int = 42):
    import random, os
    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def to_device(batch, device):
    return {k: (v.to(device) if hasattr(v, 'to') else v) for k,v in batch.items()}

def accuracy_from_logits(logits, y):
    preds = logits.argmax(dim=-1)
    return (preds == y).float().mean().item()

class AverageMeter:
    def __init__(self): self.sum=0.0; self.n=0
    def update(self, val, n=1): self.sum += val*n; self.n += n
    @property
    def avg(self): return self.sum / max(1, self.n)
