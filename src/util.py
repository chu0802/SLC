import random
import numpy as np
import torch

class LR_Scheduler(object):
    def __init__(self, optimizer, num_iters):
        self.optimizer = optimizer
        self.iter = 0
        self.num_iters = num_iters
        self.current_lr = optimizer.param_groups[-1]['lr']
    def step(self):
        self.iter += 1
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['base_lr'] * ((1 + 0.0001 * self.iter) ** (-0.75))
        self.current_lr = self.optimizer.param_groups[-1]['lr']
    def get_lr(self):
        return self.current_lr

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save(path, model):
    torch.save(model.state_dict(), path)
    
def load(path, model):
    model.load_state_dict(torch.load(path, map_location='cpu'))