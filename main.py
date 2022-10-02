import configargparse, os, random
from pathlib import Path

from ast import literal_eval
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append('src')

from model import ResModel, ProtoClassifier
from util import set_seed, save, load, LR_Scheduler
from dataset import get_all_loaders
from evaluation import evaluation, prediction
from mdh import ModelHandler, GlobalHandler

def arguments_parsing():
    p = configargparse.ArgumentParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    p.add('--config', is_config_file=True, default='./config.yaml')
    p.add('--device', type=str, default='0')
    p.add('--mode', type=str, default='ssda')
    p.add('--method', type=str, default='base')

    p.add('--dataset', type=str, default='OfficeHome')
    p.add('--source', type=int, default=0)
    p.add('--target', type=int, default=1)

    # training settings
    p.add('--seed', type=int, default=2020)
    p.add('--bsize', type=int, default=24)
    p.add('--num_iters', type=int, default=5000)
    p.add('--alpha', type=float, default=0.3)
    p.add('--beta', type=float, default=0.5)
    p.add('--gamma', type=float, default=0.9)

    p.add('--eval_interval', type=int, default=500)
    p.add('--log_interval', type=int, default=100)
    p.add('--update_interval', type=int, default=0)
    # configurations
    p.add('--dataset_cfg', type=literal_eval)
    
    # optimizer
    p.add('--lr', type=float, default=0.01)
    p.add('--momentum', type=float, default=0.9)
    p.add('--weight_decay', type=float, default=5e-4)
    p.add('--T', type=float, default=0.6)

    p.add('--note', type=str, default='')
    p.add('--init', type=str, default='')
    return p.parse_args()

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    set_seed(args.seed)

    model = ResModel('resnet34', output_dim=args.dataset['num_classes']).cuda()

    params = model.get_params(args.lr)
    opt = torch.optim.SGD(params, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    lr_scheduler = LR_Scheduler(opt, args.num_iters)

    s_train_loader, s_test_loader, t_labeled_train_loader, t_labeled_test_loader, t_unlabeled_train_loader, t_unlabeled_test_loader = get_all_loaders(args)

    if 'LC' in args.method:
        model_path = args.mdh.gh.getModelPath(args.init)
        init_model = ResModel('resnet34', output_dim=args.dataset['num_classes'])
        load(model_path, init_model)
        init_model.cuda()
        init_model.eval()

        pseudo_label, _ = prediction(t_unlabeled_test_loader, init_model)
        pseudo_label = pseudo_label.argmax(dim=1)

        ppc = ProtoClassifier(args.dataset['num_classes'], pseudo_label)
        ppc.init(model, t_unlabeled_test_loader)
    elif 'nlc' in args.method:
        model_path = args.mdh.gh.getModelPath(args.init)
        init_model = ResModel('resnet34', output_dim=args.dataset['num_classes'])
        load(model_path, init_model)
        init_model.cuda()
        init_model.eval()

        for param in init_model.parameters():
            param.requires_grad_(False)

    torch.cuda.empty_cache()

    s_iter = iter(s_train_loader)
    u_iter = iter(t_unlabeled_train_loader)
    l_iter = iter(t_labeled_train_loader)

    model.train()

    writer = SummaryWriter(args.mdh.getLogPath())
    writer.add_text('Hash', args.mdh.getHashStr())

    for i in range(1, args.num_iters+1):
        opt.zero_grad()

        sx, sy, _ = next(s_iter)
        sx, sy = sx.float().cuda(), sy.long().cuda()
        tx, ty, _ = next(l_iter)
        tx, ty = tx.float().cuda(), ty.long().cuda()

        if 'CDAC' in args.method:
            ux, _, ux1, ux2, u_idx = next(u_iter)
            ux, ux1, ux2,  u_idx = ux.float().cuda(), ux1.float().cuda(), ux2.float().cuda(), u_idx.long()
        else:  
            ux, _, u_idx = next(u_iter)
            ux, u_idx = ux.float().cuda(), u_idx.long()

        if 'LC' in args.method:
            sf = model.get_features(sx)
            sy2 = ppc(sf.detach(), args.T)
            s_loss = model.lc_loss(sf, sy, sy2, args.alpha)

            ppc.update_center(ux, u_idx, model, args.gamma)
        elif 'NL' in args.method:
            s_loss = model.nl_loss(sx, sy, args.alpha, args.T)
        elif 'nlc' in args.method:
            sy2 = F.softmax()
        else:
            s_loss = model.base_loss(sx, sy)
        
        t_loss = model.base_loss(tx, ty)
        loss = args.beta * s_loss + (1-args.beta) * t_loss

        loss.backward()
        opt.step()

        opt.zero_grad()
        if 'MME' in args.method:  
            u_loss = model.mme_loss(ux)
            u_loss.backward()
        elif 'CDAC' in args.method:
            u_loss = model.cdac_loss(ux, ux1, ux2, i)
            u_loss.backward()

        opt.step()
        lr_scheduler.step()

        if i % args.log_interval == 0:
            writer.add_scalar('LR', lr_scheduler.get_lr(), i)
            writer.add_scalar('Loss/s_loss', s_loss.item(), i)
            writer.add_scalar('Loss/t_loss', t_loss.item(), i)
            if 'MME' in args.method or 'CDAC' in args.method:
                writer.add_scalar('Loss/u_loss', -u_loss.item(), i)
        
        if i % args.eval_interval == 0:
            # s_acc = evaluation(s_test_loader, model)
            # writer.add_scalar('Acc/s_acc.', s_acc, i)
            t_acc = evaluation(t_unlabeled_test_loader, model)
            writer.add_scalar('Acc/t_acc.', t_acc, i)

        # if args.update_interval > 0 and i % args.update_interval == 0 and 'LC' in args.method:
        #     ppc = getPPC(args, model, t_unlabeled_test_loader, pseudo_label)
        
    save(args.mdh.getModelPath(), model)

if __name__ == '__main__':
    args = arguments_parsing()
    gh = GlobalHandler('test')
    args.mdh = ModelHandler(args, keys=['dataset', 'mode', 'method', 'source', 'target', 'seed', 'num_iters', 'alpha', 'T', 'init', 'note', 'update_interval', 'lr', 'gamma'], gh=gh)
    # replace the configuration
    args.dataset = args.dataset_cfg[args.dataset]
    main(args)