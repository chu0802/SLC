from itertools import permutations
from argparse import Namespace
from pathlib import Path
from mdh import GlobalHandler as GH
from numpy.random import default_rng
from random import shuffle
from shutil import rmtree

gh = GH()
mmseed = 802
dir_ = Path('script')
rmtree(str(dir_))
dir_.mkdir(parents=True, exist_ok=True)
device = [0]*12
# for DomainNet
num_domains = 4

valid_pairs = [6, 7, 3, 2, 10, 8, 4]
num_repeated = 1
# for OfficeHome
# size = 12
# valid_pairs = list(range(12))

args_list = [] 
pairs = list(permutations(range(num_domains), 2))
for mother_seed in default_rng(seed=mmseed).integers(1e4, size=num_repeated):
    
    seed_list = default_rng(seed=mother_seed).integers(1e4, size=len(valid_pairs))
    for i, (v, seed) in enumerate(zip(valid_pairs, seed_list)):
        args = Namespace()
        args.num_iters = 50000
        args.mode = 'ssda'
        args.method = 'CDAC_LC'
        args.dataset = 'DomainNet'
        args.alpha = 0.3
        args.T = 0.6
        args.lr = 0.01
        args.update_interval = 500
        args.note = f'mother_{mother_seed}'
        args.source, args.target, args.seed, args.order = *pairs[v], seed, i
        args.init = gh.regSearch(f':CDAC/.*seed:{args.seed}.*{args.source}.target.{args.target}')[0]
        args_list.append(args)
shuffle(args_list)

for i, args in enumerate(args_list):
    script_num = i % len(device)
    args.device = device[script_num]
    with (dir_ / f'scriptMME_LC{script_num}.sh').open('a') as f:
        f.write('python main.py ' + ' '.join([f'--{k} {v}' for k, v in args.__dict__.items()]) + '\n')
    with (dir_ / f'tmp.sh').open('a') as tf:
        tf.write('python test.py ' + ' '.join([f'--{k} {v}' for k, v in args.__dict__.items()]) + '\n')