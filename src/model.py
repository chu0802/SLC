from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from cdac_loss import advbce_unlabeled, sigmoid_rampup, BCE_softlabels
from evaluation import prediction
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class ProtoClassifier(nn.Module):
    def __init__(self, size, label):
        super(ProtoClassifier, self).__init__()
        self.center = None
        self.label = label
        self.size = size
    @torch.no_grad()
    def init(self, model, t_loader):
        _, t_feat = prediction(t_loader, model)
        self.center = torch.vstack([t_feat[self.label == i].mean(dim=0) for i in range(self.size)]).requires_grad_(False)
    @torch.no_grad()
    def update_center(self, x, idx, model, p=0.9):
        model.eval()

        f = model.get_features(x)
        c = torch.stack([f[self.label[idx] == i].mean(dim=0) for i in range(self.size)])
        c_idx = torch.unique(self.label[idx])
        self.center[c_idx] = p * self.center[c_idx] + (1 - p) * c[c_idx]

        model.train()
    @torch.no_grad()
    def forward(self, x, T=1.0):
        dist = torch.cdist(x, self.center)
        return F.softmax(-dist*T, dim=1)

class ResBase(nn.Module):
    def __init__(self, backbone='resnet34', **kwargs):
        super(ResBase, self).__init__()
        self.res = models.__dict__[backbone](**kwargs)
        self.last_dim = self.res.fc.in_features
        self.res.fc = nn.Identity()

    def forward(self, x):
        return self.res(x)

class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, num_classes=65, temp=0.05):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes, bias=False)
        self.temp = temp
    def forward(self, x, reverse=False):
        x = self.get_features(x, reverse=reverse)
        return self.get_predictions(x)
    def get_features(self, x, reverse=False):
        x = self.fc1(x)
        if reverse:
            x = grad_reverse(x)
        return F.normalize(x) / self.temp
    def get_predictions(self, x):
        return self.fc2(x)

class ResModel(nn.Module):
    def __init__(self, backbone='resnet34', hidden_dim=512, output_dim=65, temp=0.05, pre_trained=True):
        super(ResModel, self).__init__()
        self.f = ResBase(backbone=backbone, weights=models.__dict__[f'ResNet{backbone[6:]}_Weights'].DEFAULT if pre_trained else None)
        self.c = Classifier(self.f.last_dim, hidden_dim, output_dim, temp)
        self.c.apply(init_weights)

        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.bce = BCE_softlabels()
    def forward(self, x, reverse=False):
        return self.c(self.f(x), reverse)

    def get_params(self, lr):
        return [
            {'params': self.f.parameters(), 'base_lr': lr*0.1, 'lr': lr*0.1},
            {'params': self.c.parameters(), 'base_lr': lr, 'lr': lr}
        ]
    def get_features(self, x, reverse=False):
        return self.c.get_features(self.f(x), reverse=reverse)
    
    def get_predictions(self, x):
        return self.c.get_predictions(x)

    def base_loss(self, x, y):
        return self.criterion(self.forward(x), y).mean()
    
    def lc_loss(self, f, y1, y2, alpha):
        out = self.get_predictions(f)
        log_softmax_out = F.log_softmax(out, dim=1)
        l_loss = self.criterion(out, y1)
        soft_loss = -(y2 * log_softmax_out).sum(axis=1)
        return ((1 - alpha) * l_loss + alpha * soft_loss).mean()

    def nl_loss(self, x, y, alpha, T):
        out = self.forward(x)
        y2 = F.softmax(out.detach() * T, dim=1)
        log_softmax_out = F.log_softmax(out, dim=1)
        l_loss = self.criterion(out, y)
        soft_loss = -(y2 * log_softmax_out).sum(dim=1)
        return ((1 - alpha) * l_loss + alpha * soft_loss).mean()

    def mme_loss(self, x, lamda=0.1):
        out = self.forward(x, reverse=True)
        out = F.softmax(out, dim=1)
        return lamda * torch.mean(torch.sum(out * (torch.log(out + 1e-10)), dim=1))
    def cdac_loss(self, x, x1, x2, i):
        w_cons = 30 * sigmoid_rampup(i, 2000)

        f, f1, f2 = [self.f(i) for i in [x, x1, x2]]
        prob, prob1 = [F.softmax(self.c(i, reverse=True), dim=1) for i in [f, f1]]
        aac_loss = advbce_unlabeled(target=None, f=f, prob=prob, prob1=prob1, bce=self.bce)

        out, out1, out2 = [self.c(i) for i in [f, f1, f2]]
        prob, prob1, prob2 = [F.softmax(i, dim=1) for i in [out, out1, out2]]
        mp, pl = torch.max(prob.detach(), dim=1)
        mask = mp.ge(0.95).float()

        pl_loss = (F.cross_entropy(out2, pl, reduction='none') * mask).mean()
        con_loss = F.mse_loss(prob1, prob2)

        return aac_loss + pl_loss + w_cons * con_loss

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambd
        return output, None

def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)