import configargparse, os, random
from pathlib import Path

from ast import literal_eval
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append('src')

from model import ResModel, ProtoClassifier, ModelEMA
from util import set_seed, save, load, LR_Scheduler, Lambda_Scheduler
from dataset import get_all_loaders
from evaluation import evaluation, prediction
from mcl_loss import Prototype
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
    p.add('--shot', type=str, default='3shot', choices=['1shot', '3shot'])
    p.add('--alpha', type=float, default=0.3)
    p.add('--beta', type=float, default=0.5)

    p.add('--eval_interval', type=int, default=500)
    p.add('--log_interval', type=int, default=100)
    p.add('--update_interval', type=int, default=0)
    p.add('--early', type=int, default=5000)
    p.add('--counter', type=int, default=5)
    p.add('--warmup', type=int, default=0)
    # configurations
    p.add('--dataset_cfg', type=literal_eval)
    
    # optimizer
    p.add('--lr', type=float, default=0.01)
    p.add('--momentum', type=float, default=0.9)
    p.add('--weight_decay', type=float, default=5e-4)
    p.add('--T', type=float, default=0.6)

    p.add('--note', type=str, default='')
    p.add('--order', type=int, default=0)
    p.add('--init', type=str, default='')
    
    p.add('--from_pretrained', action='store_true', default=False)
    return p.parse_args()

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    set_seed(args.seed)

    if args.from_pretrained:
        model = ResModel('resnet34', output_dim=args.dataset['num_classes'])
        model_path = args.mdh.gh.getModelPath(args.init)
        load(model_path, model)
        model.cuda()
    else:
        model = ResModel('resnet34', output_dim=args.dataset['num_classes']).cuda()
    
    params = model.get_params(args.lr)
    opt = torch.optim.SGD(params, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    lr_scheduler = LR_Scheduler(opt, args.num_iters)
    # lr_scheduler = LR_Scheduler(opt, args.num_iters, step=50000)

    if args.mode == 'uda':
        s_train_loader, s_test_loader, t_unlabeled_train_loader, t_unlabeled_test_loader = get_all_loaders(args)
    elif args.mode == 'ssda':
        s_train_loader, s_test_loader, t_labeled_train_loader, t_labeled_test_loader, t_unlabeled_train_loader, t_unlabeled_test_loader, t_val_loader = get_all_loaders(args)

    if 'LC' in args.method:
        # model_path = args.mdh.gh.getModelPath(args.init)
        # init_model = ResModel('resnet34', output_dim=args.dataset['num_classes'])
        # load(model_path, init_model)
        # init_model.cuda()

        # pseudo_label, _ = prediction(t_unlabeled_test_loader, init_model)
        # pseudo_label = pseudo_label.argmax(dim=1)
        # init_model.eval()

        # ppc = ProtoClassifier(args.dataset['num_classes'], pseudo_label)
        ppc = ProtoClassifier(args.dataset['num_classes'])
        if args.warmup == 0:
            if 'ideal' in args.method:
                ppc.ideal_init(model, t_unlabeled_test_loader)
            else:
                ppc.init(model, t_unlabeled_test_loader)

    if 'MCL' in args.method:
        proto = Prototype(C=args.dataset['num_classes'], dim=512)

    torch.cuda.empty_cache()

    s_iter = iter(s_train_loader)
    u_iter = iter(t_unlabeled_train_loader)

    if args.mode == 'ssda':
        l_iter = iter(t_labeled_train_loader)

    model.train()

    writer = SummaryWriter(args.mdh.getLogPath())
    writer.add_text('Hash', args.mdh.getHashStr())

    counter = 0
    best_acc, best_ema_acc, best_val_acc = 0, 0, 0

    if 'MCL' in args.method:
        model_ema = ModelEMA(model, decay=0.99)

    # for i in range(50001, 100001):
    for i in range(1, args.num_iters+1):
        opt.zero_grad()

        sx, sy, _ = next(s_iter)
        sx, sy = sx.float().cuda(), sy.long().cuda()

        if 'CDAC' in args.method or 'MCL' in args.method or 'PL' in args.method:
            ux, uy, ux1, ux2, u_idx = next(u_iter)
            ux, ux1, ux2, u_idx = ux.float().cuda(), ux1.float().cuda(), ux2.float().cuda(), u_idx.long()
            # for testing only
            uy = uy.long().cuda()
        else:  
            ux, _, u_idx = next(u_iter)
            ux, u_idx = ux.float().cuda(), u_idx.long()

        sf = model.get_features(sx)

        if i > args.warmup and 'LC' in args.method:
            sy2 = ppc(sf.detach(), args.T)
            s_loss = model.lc_loss(sf, sy, sy2, args.alpha)
        elif 'NL' in args.method:
            s_loss = model.nl_loss(sf, sy, args.alpha, args.T)
        else:
            s_loss = model.feature_base_loss(sf, sy)
        
        if args.mode == 'uda':
            loss = s_loss
        elif args.mode == 'ssda':
            tx, ty, _ = next(l_iter)
            tx, ty = tx.float().cuda(), ty.long().cuda()
            t_loss = model.base_loss(tx, ty)
            loss = args.beta * s_loss + (1-args.beta) * t_loss

        loss.backward()
        opt.step()

        opt.zero_grad()
        if 'MME' in args.method:  
            u_loss = model.mme_loss(ux)
            u_loss.backward()
        elif 'CDAC' in args.method:
            u_loss, num_pl, ratio_pl, acc_pl = model.cdac_loss(ux, uy, ux1, ux2, i)
            u_loss.backward()
        elif 'PL' in args.method:
            u_loss = model.pl_loss(ux, ux1, ux2)
            u_loss.backward()
        elif 'MCL' in args.method:
            u_loss = model.mcl_loss(ux, ux1, proto, i, args.dataset['num_classes'])
            u_loss.backward()

            proto.update(sf, sy, i, norm=True)

        opt.step()

        if 'LC' in args.method and args.warmup > 0 and i == args.warmup:
            ppc.init(model, t_unlabeled_test_loader)
            lr_scheduler.refresh()
        lr_scheduler.step()

        if 'MCL' in args.method:
            model_ema.update(model)

        if i % args.log_interval == 0:
            writer.add_scalar('LR', lr_scheduler.get_lr(), i)
            writer.add_scalar('Loss/s_loss', s_loss.item(), i)
            if args.mode == 'ssda':
                writer.add_scalar('Loss/t_loss', t_loss.item(), i)
            if 'MME' in args.method or 'CDAC' in args.method or 'MCL' in args.method or 'PL' in args.method:
                writer.add_scalar('Loss/u_loss', -u_loss.item(), i)

            if 'CDAC' in args.method:
                writer.add_scalar('PseudoLabel/numbers', num_pl, i)
                writer.add_scalar('PseudoLabel/ratio', ratio_pl, i)
                writer.add_scalar('PseudoLabel/acc', acc_pl, i)
        if i >= args.early and i % args.eval_interval == 0:
            val_acc = evaluation(t_val_loader, model)
            writer.add_scalar('Acc/val_acc', val_acc, i)
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                counter = 0

                t_acc = evaluation(t_unlabeled_test_loader, model)
                writer.add_scalar('Acc/t_acc', t_acc, i)
                save(args.mdh.getModelPath(), model)
                best_acc = t_acc

                if 'MCL' in args.method:
                    ema_acc = evaluation(t_unlabeled_test_loader, model_ema.ema)
                    writer.add_scalar('Acc/ema_acc', ema_acc, i)
                    best_ema_acc = ema_acc
            else:
                counter += 1
            if counter > args.counter or i == args.num_iters:
                writer.add_scalar('Acc/final_acc', best_acc, 0)
                if 'MCL' in args.method:
                    writer.add_scalar('Acc/final_ema_acc', best_ema_acc, 0)
                break
    
        if i > args.warmup and args.update_interval > 0 and i % args.update_interval == 0 and 'LC' in args.method:
            if 'ideal' in args.method:
                ppc.ideal_init(model, t_unlabeled_test_loader)
            else:
                ppc.init(model, t_unlabeled_test_loader)
    # for saving pre-trained model
    # t_acc = evaluation(t_unlabeled_test_loader, model)
    # writer.add_scalar('Acc/t_acc', t_acc, args.num_iters)
    # save(args.mdh.getModelPath(), model)

if __name__ == '__main__':
    args = arguments_parsing()
    gh = GlobalHandler('base')
    args.mdh = ModelHandler(args, keys=['dataset', 'mode', 'method', 'source', 'target', 'seed', 'num_iters', 'alpha', 'T', 'init', 'note', 'update_interval', 'lr', 'order', 'shot', 'warmup'], gh=gh)
    # replace the configuration
    args.dataset = args.dataset_cfg[args.dataset]
    main(args)