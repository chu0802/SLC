import random
import numpy as np
import torch

class LR_Scheduler(object):
    def __init__(self, optimizer, num_iters, step=0, final_lr=None):
        # if final_lr: use cos scheduler, otherwise, use gamma scheduler
        self.final_lr = final_lr
        self.optimizer = optimizer
        self.iter = step
        self.num_iters = num_iters
        self.current_lr = optimizer.param_groups[-1]['lr']
    def step(self):
        for param_group in self.optimizer.param_groups:
            base = param_group['base_lr']
            self.current_lr = param_group['lr'] = (
                self.final_lr + 0.5 * (base - self.final_lr)*(1 + np.cos(np.pi * self.iter/self.num_iters))
                if self.final_lr
                else base * ((1 + 0.0001 * self.iter) ** (-0.75))
                # else base * ((1 + 5 * self.iter / self.num_iters) ** (-0.75))
            )
        self.iter += 1
    def refresh(self):
        self.iter = 0
    def get_lr(self):
        return self.current_lr

class Lambda_Scheduler(object):
    def __init__(self, num_iters, base_lambda=0.01, propotion=True):
        self.iter = 0
        self.num_iters = num_iters
        self.propotion = propotion
        self.base = base_lambda
        self.current_lambda = base_lambda
    def step(self):
        if self.propotion:
            self.current_lambda = self.base * ((1 + 5 * self.iter / self.num_iters)**(-0.75))
        else:
            self.current_lambda = self.base * ((1 + 0.0001 * self.iter) ** (-0.75))
        self.iter += 1
    def get_lambda(self):
        return self.current_lambda

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